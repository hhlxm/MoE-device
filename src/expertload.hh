#pragma once
#include <fcntl.h>
#include <ggml.h>
#include <liburing.h>
#include <llama.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <queue>
#include <regex>
#include <vector>

#include "llama-hparams.h"
#include "llama-impl.h"
#include "llama-mmap.h"
#include "llama-arch.h"

inline bool is_file_empty(const std::string& filename) {
    return std::filesystem::exists(filename) && std::filesystem::file_size(filename) == 0;
}


enum expert_weight {
    EXPERT_NONE = -1,  // 未定义
    EXPERT_GATE = 0,   // gate
    EXPERT_UP   = 1,   // up
    EXPERT_DOWN = 2    // down
};

struct ExpertIOContext {
    void *        dst;         // 目标缓冲区
    std::vector<int> expert_ids;  // 专家ID列表
    int              layer_id;    // 层ID
    int              m;           // 专家类型（0: gate, 1: up, 2: down）
    size_t           n_size;      // 专家数据大小
};

struct ExpertOffset {
    size_t offs;    // tensor data offset in the original file
    size_t n_size;  // size of the experts data
};

class expert_loader {
  private:
    const int                     NUM_IO = 512;                                   // IO队列深度
    struct io_uring               ring;                                           // io_uring实例
    std::queue<ExpertIOContext *> ctx_queue;
    size_t                        buffer_size = 0;                                // 读缓冲区大小
    int                           on_fly      = 0;                                // 正在进行的IO数量
    bool                          initialized;                                    // 是否已初始化
    char                          save_path[128] = "/tmp/expert.bin";  // 保存文件路径
    char                          log_path[128] = "/tmp/expert_log.log";  // 保存文件路径
    char                          padding[4096] ;

    // 文件相关
    std::vector<int> files;               // 文件列表
    size_t           alignment;           // 对齐要求
    bool f_write = false;

    int    save_fd          = -1;         // 保存文件的文件描述符
    size_t save_offset      = 0;          // 保存文件的当前偏移量
    bool   save_initialized = false;      // 是否已初始化保存文件

    // 模型参数
    int num_layers;   // 层数
    int num_experts;  // 每层专家数

    // 性能统计
    std::atomic<uint64_t> g_load_cnt{ 0 };
    std::atomic<uint64_t> g_load_size{ 0 };
    std::atomic<uint64_t> g_load_time{ 0 };
    std::atomic<uint64_t> g_load_layers_time{ 0 };

    // 每层专家数据的基础偏移量
    std::vector<std::vector<ExpertOffset>> offsets;

    // 私有构造函数
    expert_loader() : initialized(false) {
        memset(&ring, 0, sizeof(ring));
        // 初始化io_uring
        if (io_uring_queue_init(256, &ring, 0) < 0) {
            fprintf(stderr, "[Expert Loader] Failed to initialize io_uring\n");
            initialized = false;
        } else {
            initialized = true;
        }
    }

    // 删除拷贝构造和赋值操作符
    expert_loader(const expert_loader &)             = delete;
    expert_loader & operator=(const expert_loader &) = delete;


  public:
    static expert_loader & get_instance() {
        static expert_loader instance;
        return instance;
    }

    ~expert_loader() {
        if (initialized) {
            io_uring_queue_exit(&ring);
        }
        if (save_fd >= 0) {
            close(save_fd);
            save_fd = -1;
        }

        while (!ctx_queue.empty()) {
            delete ctx_queue.front();
            ctx_queue.pop();
        }
    }

    // 从model初始化
    bool init(const llama_hparams & hparams, const std::string & name) {
        this->num_layers  = hparams.n_layer;
        this->num_experts = hparams.n_expert;
        this->alignment   = 4096;//TENSOR_ALIGNMENT

        memset(padding, 0, 4096);  // 填充为0

        this->offsets.resize(num_layers);
        for (int i = 0; i < num_layers; i++) {
            offsets[i].resize(num_experts);
        }
        this->buffer_size = 0;
        for (int i = 0; i < NUM_IO; i++) {
            auto * ctx    = new ExpertIOContext();
            ctx_queue.push(ctx);
        }

        this->initialized = true;

        std::string save_temp = "/home/wangtuowei/lxm/temp/moe_expert_" + name + ".bin";
        strncpy(save_path, save_temp.c_str(), sizeof(save_path) - 1);
        save_path[sizeof(save_path) - 1] = '\0'; // 确保字符串以null结尾

        
        std::string save_log ="/home/wangtuowei/lxm/temp/moe_expert_" + name + "_log.log";
        strncpy(log_path, save_log.c_str(), sizeof(log_path) - 1);
        log_path[sizeof(log_path) - 1] = '\0'; // 确保字符串以null结尾

        
        // this->save_fd = open(save_path, O_RDWR | O_CREAT | O_DIRECT, 0666);
        // if (save_fd <= 0) {
        //     std::cerr << "open error: " << save_path << std::endl;
        //     return false;
        // }
      
        return true;
    }

    void clear_file(const char * path) {
        FILE * fp = fopen(path, "w");
        if (fp) {
            fclose(fp);
        }
    }

    bool is_initialized() const { return initialized; }

    // 读取单层单个专家
    bool read_expert(int layer_id, int expert_id, void * dst) {
        std::vector<int>       expert_ids = { expert_id };
        std::vector<void *> dsts       = { dst };
        return read_experts(layer_id, expert_ids, dsts);
    }

    // 读取单层多个专家
    bool read_experts(int layer_id, std::vector<int> & expert_ids, const std::vector<void *> & dsts) {
        this->save_fd = open(save_path, O_RDONLY  | O_DIRECT, 0666);
        if (save_fd < 0) {
            fprintf(stderr, "[Expert Loader] Save file not opened %d\n",save_fd);
            return false;
        }
        bool ok         = true;
        int  submit_num = 0;

        std::sort(expert_ids.begin(), expert_ids.end());
        expert_ids.erase(std::unique(expert_ids.begin(), expert_ids.end()), expert_ids.end());
        // auto start      = std::chrono::high_resolution_clock::now();

        // 处理连续的expert_ids组
        int p1   = 0;
        int size = expert_ids.size();
        // printf("[Expert Loader] Read layer %d experts %ld\n", layer_id, size);
        int idx  = 0;
        while (p1 < size) {
            int p2 = p1;
            // 找到连续的专家ID
            while (p2 + 1 < size && expert_ids[p2 + 1] == expert_ids[p2] + 1) {
                p2++;
            }

            // 等待IO槽位可用
            while (on_fly >= NUM_IO) {
                ok = ok && process_completion();
            }

            // 提交读请求
            std::vector<int>                 batch_ids(expert_ids.begin() + p1, expert_ids.begin() + p2 + 1);
            std::vector<std::pair<int, int>> batch_info;
            for (auto it : batch_ids) {
                // printf("batch_ids: %d \n ", it);
                // GGML_ASSERT(idx >= 0 && idx < 6);
                batch_info.push_back(std::make_pair(idx++, it));
            }

            for (int i = 0; i < (int)dsts.size(); i++) {
                ok = ok && submit_read_request(layer_id, batch_info, dsts[i], i);
                submit_num++;
            }

            p1 = p2 + 1;
        }
        // // 等待所有IO完成
        // while (on_fly > 0) {
        //     ok = ok && process_completion();
        // }
        // auto end = std::chrono::high_resolution_clock::now();

        // auto during = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
        // g_load_time += during;
        // auto temp =(unsigned long long)g_load_size.load();
        // printf("[Expert Loader] size:%llu time: %.6fms\n", temp, (unsigned long long)during/1000.0);

        return ok;
    }

    // 读取多层多个专家
    bool read_experts_multi_layer(const std::vector<int> & layer_ids, std::vector<std::vector<int>> & expert_ids,
                                  const std::vector<std::vector<void *>> & dsts) {
        bool ok = true;
        for (size_t i = 0; i < layer_ids.size(); i++) {
            ok = ok && read_experts(layer_ids[i], expert_ids[i], dsts[i]);
        }
        return ok;
    }

    bool is_save_initialized() const { return save_initialized; }


    bool write_experts(int layer_id, ggml_tensor * tensor, int m) {
        this->save_fd = open(save_path, O_WRONLY | O_CREAT , 0666);
        if (!is_initialized()) {
            fprintf(stderr, "[Expert Loader] Not initialized\n");
            return false;
        }

        if (save_fd < 0) {
            fprintf(stderr, "[Expert Loader] Save file not opened %d\n",save_fd);
            return false;
        }

        if (layer_id < 0 || layer_id >= num_layers) {
            fprintf(stderr, "[Expert Loader] Invalid layer ID: %d\n", layer_id);
            return false;
        }


        if(!is_file_empty(save_path)&&!f_write)
        {
            std::ifstream log_file(log_path);
            std::string   line;
            std::regex    pattern(R"(Write layer (\d+) expert (\d+),.*aligned size (\d+) offset (\d+))");
            while (std::getline(log_file, line)) {
                std::smatch match;
                if (std::regex_search(line, match, pattern)) {
                    int    layer        = std::stoi(match[1]);
                    int    expert       = std::stoi(match[2]);
                    size_t aligned_size = std::stoll(match[3]);
                    size_t offset       = std::stoll(match[4]);
                    if (layer == layer_id && expert == m) {
                        // std::cout << "layer: " << layer
                        //         << ", expert: " << expert
                        //         << ", aligned_size: " << aligned_size
                        //         << ", offset: " << offset << std::endl;
                        // 你可以在这里处理数据，比如赋值给 offsets[layer_id][m]
                        offsets[layer_id][m] = ExpertOffset{ offset, aligned_size / this->num_experts };
                        break;  // 找到后退出循环
                    }
                }
            }
            return true;
        }
        else
        {
            FILE * log_file = fopen(log_path, "a");
            if(log_file == nullptr) {
                fprintf(stderr, "[Expert Loader] Failed to open log file: %s\n", strerror(errno));
                return false;
            }
            f_write=true;

            // 获取base 和 pad
            const char * base_ptr = (char *)tensor->data;
            size_t n_size = ggml_nbytes(tensor);    //tensor的原始大小
            size_t per_size = n_size / tensor->ne[2];  // 每个专家的大小
            size_t per_aligned_size =GGML_PAD(per_size,alignment);//每个专家的对齐大小
            size_t pad_size = per_aligned_size - per_size;  // 填充大小

            fprintf(log_file,"[Expert Loader] Write layer %d expert %d, tensor size %zu, aligned size %zu offset %zu\n", layer_id, m, n_size, per_aligned_size * tensor->ne[2], save_offset);
            GGML_ASSERT(save_offset % alignment == 0);  // 确保对齐
            offsets[layer_id][m] = ExpertOffset{save_offset, per_aligned_size};

            // 每个expert pad写
            for(int i=0;i<tensor->ne[2];i++)
            {
                ssize_t bytes_written = pwrite(save_fd, base_ptr+per_size*i, per_size, save_offset);
                if (bytes_written < 0) {
                        fprintf(stderr, "[Expert Loader] Failed to write to save file: %s\n", strerror(errno));
                        return false;
                    }
                GGML_ASSERT(bytes_written == (ssize_t)per_size);
                save_offset += bytes_written;
                if(pad_size != 0)
                {
                    ssize_t bytes_written = pwrite(save_fd, padding, pad_size, save_offset);
                    if (bytes_written < 0) {
                        fprintf(stderr, "[Expert Loader] Failed to write to save file: %s\n", strerror(errno));
                        return false;
                    }
                    GGML_ASSERT(bytes_written == (ssize_t)pad_size);
                    save_offset += bytes_written;
                }
                GGML_ASSERT(save_offset % alignment == 0);  // 确保对齐
            }

            fclose(log_file);
            return true;
        }
    }

    // 获取性能统计信息
    std::tuple<uint64_t, uint64_t, uint64_t> get_io_stats() {
        return { g_load_cnt.load(), g_load_size.load(), g_load_time.load() };
    }

    bool submit_read_request(int layer_id, const std::vector<std::pair<int, int>> & batch_info,  void * dst, int m) {
        struct ExpertOffset & offset      = offsets[layer_id][m];
        size_t                read_size   = offset.n_size * batch_info.size();//对齐的大小
        size_t                read_offset = offset.offs + offset.n_size * batch_info[0].second;
        void *              dst_ptr     = (char *)dst + offset.n_size * batch_info[0].first;
        GGML_ASSERT(((uintptr_t)dst) % alignment == 0);           // dst 必须对齐
        GGML_ASSERT(((uintptr_t)dst_ptr) % alignment == 0);       // dst_ptr 必须对齐
        GGML_ASSERT(read_size % alignment == 0);                  // 读大小必须对齐
        GGML_ASSERT(read_offset % alignment == 0);                // 偏移必须对齐
        // 可选 - 检查文件大小是否足够（假设 save_fd 仍然打开）
        off_t file_size = lseek(save_fd, 0, SEEK_END);
        GGML_ASSERT(read_offset + read_size <= (size_t)file_size);


        // 获取IO上下文
        ExpertIOContext * ctx = ctx_queue.front();
        ctx_queue.pop();

        // ctx->expert_ids = NULL;
        ctx->dst      = dst_ptr;
        ctx->layer_id = layer_id;
        ctx->m        = m;
        ctx->n_size   = offset.n_size;

        // 提交IO请求
        struct io_uring_sqe * sqe = io_uring_get_sqe(&ring);
        if (!sqe) {
            fprintf(stderr, "[Expert Loader] Failed to get sqe\n");
            return false;
        }

        io_uring_prep_read(sqe, save_fd, dst_ptr, read_size, read_offset);
        io_uring_sqe_set_data(sqe, ctx);

        if (io_uring_submit(&ring) < 0) {
            fprintf(stderr, "[Expert Loader] Failed to submit IO request\n");
            return false;
        }

        on_fly++;
        g_load_cnt++;
        g_load_size += read_size;
        return true;
    }

    bool process_completion() {
        struct io_uring_cqe * cqe;
        int                   ret = io_uring_wait_cqe(&ring, &cqe);

        if (ret < 0) {
            fprintf(stderr, "[Expert Loader] Error waiting for completion: %d\n", ret);
            return false;
        }

        ExpertIOContext * ctx = static_cast<ExpertIOContext *>(io_uring_cqe_get_data(cqe));
        if (cqe->res <= 0) {
            // fprintf(stderr, "[Expert Loader] Read failed: %s %d \n", strerror(-cqe->res), cqe->res);
                GGML_ABORT("[Expert Loader] Read failed: %s %d \n", strerror(-cqe->res), cqe->res);
        }

        io_uring_cqe_seen(&ring, cqe);
        ctx_queue.push(ctx);
        on_fly--;
        return true;
    }

    void wait_upload_finish(int flag) {
        auto start = std::chrono::high_resolution_clock::now();
        while (on_fly > 0) {
            process_completion();
        }
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
        if(flag == 0)
        {
            g_load_layers_time += duration;//prefill 异步加载
        }
        else if (flag == 1) {
            g_load_time += duration;//expert加载
        }
        
    }
};

inline bool parse_blk_ffn_exps(const char * str, int & layer_id, expert_weight & m) {
    // 将 char* 转换为 std::string
    std::string input(str);

    // 定义gate
    std::regex  pattern_gate(R"(blk\.(\d+)\.ffn_gate_exps\.weight)");
    // 定义up
    std::regex  pattern_up(R"(blk\.(\d+)\.ffn_up_exps\.weight)");
    // 定义down
    std::regex  pattern_down(R"(blk\.(\d+)\.ffn_down_exps\.weight)");
    std::smatch match;

    // 匹配字符串并提取数字
    if (std::regex_match(input, match, pattern_gate)) {
        layer_id = std::stoi(match[1].str());
        m        = EXPERT_GATE;  // gate
        return true;
    } else if (std::regex_match(input, match, pattern_up)) {
        layer_id = std::stoi(match[1].str());
        m        = EXPERT_UP;  // up
        return true;
    } else if (std::regex_match(input, match, pattern_down)) {
        layer_id = std::stoi(match[1].str());
        m        = EXPERT_DOWN;  // down
        return true;
    }
    return false;
}

inline bool parse_blk_ffn_exps(const char * str, int & layer_id, int & m) {
    // 将 char* 转换为 std::string
    std::string input(str);

    // 定义gate
    std::regex  pattern_gate(R"(blk\.(\d+)\.ffn_gate_exps\.weight)");
    // 定义up
    std::regex  pattern_up(R"(blk\.(\d+)\.ffn_up_exps\.weight)");
    // 定义down
    std::regex  pattern_down(R"(blk\.(\d+)\.ffn_down_exps\.weight)");
    std::smatch match;

    // 匹配字符串并提取数字
    if (std::regex_match(input, match, pattern_gate)) {
        layer_id = std::stoi(match[1].str());
        m        = 0;  // gate
        return true;
    } else if (std::regex_match(input, match, pattern_up)) {
        layer_id = std::stoi(match[1].str());
        m        = 1;  // up
        return true;
    } else if (std::regex_match(input, match, pattern_down)) {
        layer_id = std::stoi(match[1].str());
        m        = 2;  // down
        return true;
    }
    return false;
}
