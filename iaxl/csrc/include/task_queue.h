// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <condition_variable>
#include <deque>
#include <functional>
#include <future>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <pthread.h>
#include <sched.h>

#include "env.h"
#include "iaxl_common.h"

class TaskQueue {
  public:
    static constexpr int PRIORITY_HIGH = 0;
    static constexpr int PRIORITY_LOW = 1;

    explicit TaskQueue(const char *name) : name_(name) {}

    ~TaskQueue() { shutdown(); }

    void init() {
        std::lock_guard<std::mutex> lock(mutex_);
        if (state_ == State::RUNNING)
            return;
        if (state_ != State::CREATED) {
            throw std::runtime_error("TaskQueue cannot be restarted after shutdown");
        }

        worker_ = std::thread([this]() {
            pthread_setname_np(pthread_self(), name_);
            bind_cpus();
            worker_loop();
        });
        state_ = State::RUNNING;
    }

    template <typename F> std::future<void> submit(F &&func, int priority = PRIORITY_HIGH) {
        auto task = std::make_shared<std::packaged_task<void()>>(std::forward<F>(func));
        auto future = task->get_future();

        {
            std::lock_guard<std::mutex> lock(mutex_);
            if (state_ != State::RUNNING) {
                throw std::runtime_error("TaskQueue submit rejected: queue is not running");
            }
            if (priority == PRIORITY_HIGH) {
                high_priority_tasks_.push_back([task]() { (*task)(); });
            } else if (priority == PRIORITY_LOW) {
                low_priority_tasks_.push_back([task]() { (*task)(); });
            } else {
                throw std::invalid_argument(
                    "TaskQueue priority must be PRIORITY_HIGH or PRIORITY_LOW");
            }
        }
        cv_.notify_one();

        return future;
    }

    void shutdown() {
        {
            std::unique_lock<std::mutex> lock(mutex_);
            if (state_ == State::CREATED) {
                state_ = State::STOPPED;
                return;
            }
            if (state_ == State::STOPPED)
                return;
            if (state_ == State::STOPPING) {
                state_cv_.wait(lock, [this]() { return state_ == State::STOPPED; });
                return;
            }
            state_ = State::STOPPING;
        }
        cv_.notify_one();

        if (worker_.joinable()) {
            worker_.join();
        }

        {
            std::lock_guard<std::mutex> lock(mutex_);
            state_ = State::STOPPED;
        }
        state_cv_.notify_all();
    }

    // Pins to IAXL_CPU_OMP_THREADS_BIND ("0-7,16"); threads spawned afterwards inherit it.
    static void bind_cpus() {
        const char *spec = env_str("IAXL_CPU_OMP_THREADS_BIND", nullptr);
        if (!spec)
            return;
        cpu_set_t cpus;
        CPU_ZERO(&cpus);
        for (const char *p = spec;; ++p) {
            char *end;
            long first = strtol(p, &end, 10);
            long last = first;
            if (end != p && *end == '-')
                last = strtol(p = end + 1, &end, 10);
            IAXL_CHECK(end != p && 0 <= first && first <= last && last < CPU_SETSIZE &&
                           (*end == ',' || *end == '\0'),
                       "invalid IAXL_CPU_OMP_THREADS_BIND");
            for (long cpu = first; cpu <= last; ++cpu)
                CPU_SET(cpu, &cpus);
            if (*end == '\0')
                break;
            p = end;
        }
        IAXL_CHECK(pthread_setaffinity_np(pthread_self(), sizeof(cpus), &cpus) == 0,
                   "cannot bind to IAXL_CPU_OMP_THREADS_BIND");
    }

  private:
    enum class State { CREATED, RUNNING, STOPPING, STOPPED };

    void worker_loop() {
        while (true) {
            std::function<void()> task;

            {
                std::unique_lock<std::mutex> lock(mutex_);
                cv_.wait(lock, [this]() {
                    return state_ != State::RUNNING || !high_priority_tasks_.empty() ||
                           !low_priority_tasks_.empty();
                });

                if (state_ == State::STOPPING && high_priority_tasks_.empty() &&
                    low_priority_tasks_.empty()) {
                    return;
                }

                if (!high_priority_tasks_.empty()) {
                    task = std::move(high_priority_tasks_.front());
                    high_priority_tasks_.pop_front();
                } else {
                    task = std::move(low_priority_tasks_.front());
                    low_priority_tasks_.pop_front();
                }
            }

            task();
        }
    }

    const char *name_;
    std::thread worker_;
    std::mutex mutex_;
    std::condition_variable cv_;
    std::condition_variable state_cv_;
    std::deque<std::function<void()>> high_priority_tasks_;
    std::deque<std::function<void()>> low_priority_tasks_;
    State state_ = State::CREATED;
};
