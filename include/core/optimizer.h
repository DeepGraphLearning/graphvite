/**
 * Copyright 2019 MilaGraph. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * @author Zhaocheng Zhu
 */

#pragma once

#include <string>
#include <functional>

namespace graphvite {

/** The type of optimizer is used for compile-time binding in GPU code */
enum OptimizerType {
    kSGD = 0,
    kMomentum,
    kAdaGrad,
    kRMSprop,
    kAdam,
    kNumOptimizer // for convenience
};

/**
 * @brief Learning rate schedule
 *
 * The learning rate schedule class is used for compatibility of string schedule and custom schedule function.
 * Generally, you don't need to construct this class explictly.
 */
class LRSchedule {
public:
    typedef std::function<float(int, int)> ScheduleFunction;

    std::string schedule;
    ScheduleFunction schedule_function;

    /** Construct a learning rate schedule */
    LRSchedule(const std::string &_schedule = "constant") : schedule(_schedule) {
        CHECK(schedule == "linear" || schedule == "constant") << "Invalid schedule `" << schedule << "`";
        if (schedule == "linear")
            schedule_function = linear_schedule;
        if (schedule == "constant")
            schedule_function = constant_schedule;
    }

    /** Construct a learning rate from custom function */
    LRSchedule(const ScheduleFunction &_schedule_function) :
            schedule("custom"), schedule_function(_schedule_function) {}

    LRSchedule(const char *_schedule) : LRSchedule(std::string(_schedule)) {}

    /** Call the schedule function */
    float operator ()(int batch_id, int num_batch) {
        return schedule_function(batch_id, num_batch);
    }

    /** Return information about the schedule */
    std::string info() const {
        std::stringstream ss;
        ss << "lr schedule: " << schedule;
        return ss.str();
    }

    /** Linear decay schedule */
    static float linear_schedule(int batch_id, int num_batch) {
        return std::max(1 - float(batch_id) / num_batch, 1e-4f);
    }

    /** Constant schedule */
    static float constant_schedule(int batch_id, int num_batch) {
        return 1;
    }
};

/**
 * @brief General interface of first-order optimizers
 *
 * The optimizer class is implemented in a way to trade off compile-time binding and run-time binding.
 *
 * In CPU code, optimizers are binded at run time to reduce the size of executables.
 * In GPU code, optimizers are binded at compile time to maximize the run-time speed.
 *
 * @note To add a new optimizer, you need to
 * - add a value in OptimizerType
 * - implement an update function in Optimizer
 * - implement a helper class for that optimizer
 * - instantiate kernels with the optimizer in Worker::kernel_dispatch()
 * - add python binding of the helper class in bind.h & graphvite.cu
 */
class Optimizer {
public:
    int num_moment;
    std::string type;
    float init_lr, lr, weight_decay;
    LRSchedule schedule;
    // auxiliary fields for different optimizers
    union {
        struct { // Momentum
            float momentum;
        };
        struct { // RMSprop
            float alpha;
        };
        struct { // Adam
            float beta1, beta2;
        };
    };
    float epsilon;

    /** Construct a default optimizer */
    Optimizer(int _type) : type("Default"), init_lr(0), lr(0) {
        CHECK(_type == kAuto) << "Only kAuto can be used for initializing a default optimizer. "
                              << "Please use a float value if you want to specify the learning rate.";
    }

    /** Construct a default optimizer with learning rate */
    Optimizer(float _lr = 1e-4) : type("Default"), init_lr(_lr), lr(_lr) {}

    /** Compute current learning rate according to the schedule */
    void apply_schedule(int batch_id, int num_batch) {
        lr = init_lr * schedule(batch_id, num_batch);
    }

    /** Return information about the optimizer */
    std::string info() const {
        std::stringstream ss;
        ss << "optimizer: " << type << std::endl;
        ss << "learning rate: " << init_lr << ", " << schedule.info() << std::endl;
        ss << "weight decay: " << weight_decay;

        if (type != "Default" && type != "SGD") {
            ss << std::endl;
            if (type == "Momentum")
                ss << "momentum: " << momentum;
            if (type == "AdaGrad")
                ss << "epsilon: " << epsilon;
            if (type == "RMSprop")
                ss << "alpha: " << alpha << ", epsilon: " << epsilon;
            if (type == "Adam")
                ss << "beta1: " << beta1 << ", beta2: " << beta2 << ", epsilon: " << epsilon;
        }
        return ss.str();
    }

    /**
     * @brief SGD update rule
     * @tparam Float floating type of parameters
     */
    template<class Float>
    __device__ inline Float sgd_update(Float parameter, Float gradient, Float weight = 1) const {
        return lr * weight * (gradient + weight_decay * parameter);
    }

    /**
     * @brief Momentum update rule
     * @tparam Float floating type of parameters
     */
    template<class Float>
    __device__ inline Float momentum_update(Float parameter, Float gradient, Float &moment1, Float weight = 1) const {
        Float regularized = weight * (gradient + weight_decay * parameter);
        moment1 = momentum * moment1 + (1 - momentum) * regularized;
        return lr * moment1;
    }

    /**
     * @brief AdaGrad update rule
     * @tparam Float floating type of parameters
     */
    template<class Float>
    __device__ inline Float adagrad_update(Float parameter, Float gradient, Float &moment1, Float weight = 1) const {
        Float regularized = weight * (gradient + weight_decay * parameter);
        moment1 += regularized * regularized;
        return lr * regularized / (sqrt(moment1) + epsilon);
    }

    /**
     * @brief RMSprop update rule
     * @tparam Float floating type of parameters
     */
    template<class Float>
    __device__ inline Float rmsprop_update(Float parameter, Float gradient, Float &moment1, Float weight = 1) const {
        Float regularized = weight * (gradient + weight_decay * parameter);
        moment1 = alpha * moment1 + (1 - momentum) * regularized * regularized;
        return lr * regularized / sqrt(moment1 + epsilon);
    }

    /**
     * @brief Adam update rule
     * @tparam Float floating type of parameters
     */
    template<class Float>
    __device__ inline Float adam_update(Float parameter, Float gradient, Float &moment1, Float &moment2,
            Float weight = 1) const {
        Float regularized = weight * (gradient + weight_decay * parameter);
        moment1 = beta1 * moment1 + (1 - beta1) * regularized;
        moment2 = beta2 * moment2 + (1 - beta2) * regularized * regularized;
        return lr *  moment1 / (sqrt(moment2) + epsilon);
    }

protected:
    Optimizer(const std::string &_type, int _num_moment = 0, float _lr = 0.025, float _weight_decay = 0,
              const LRSchedule &_schedule = "linear") :
            type(_type), init_lr(_lr), lr(_lr), weight_decay(_weight_decay), num_moment(_num_moment),
            schedule(_schedule) {}
};

/**
 * @brief Compile-time binding of 0-moment optimizers
 * @tparam Float floating type of parameters
 * @tparam type type of optimizer
 * @return the update function of the optimizer
 */
template<class Float, OptimizerType type>
__device__ decltype(&Optimizer::sgd_update<Float>) get_update_function() {
    switch (type) {
        case kSGD:
            return &Optimizer::sgd_update<Float>;
        default:
            return nullptr;
    }
}

/**
 * @brief Compile-time binding of 1-moment optimizers
 * @tparam Float floating type of parameters
 * @tparam type type of optimizer
 * @return the update function of the optimizer
 */
template<class Float, OptimizerType type>
__device__ decltype(&Optimizer::momentum_update<Float>) get_update_function_1_moment() {
    switch (type) {
        case kMomentum:
            return &Optimizer::momentum_update<Float>;
        case kAdaGrad:
            return &Optimizer::adagrad_update<Float>;
        case kRMSprop:
            return &Optimizer::rmsprop_update<Float>;
        default:
            return nullptr;
    }
}

/**
 * @brief Compile-time binding of 2-moment optimizers
 * @tparam Float floating type of parameters
 * @tparam type type of optimizer
 * @return the update function of the optimizer
 */
template<class Float, OptimizerType type>
__device__ decltype(&Optimizer::adam_update<Float>) get_update_function_2_moment() {
    switch (type) {
        case kAdam:
            return &Optimizer::adam_update<Float>;
        default:
            return nullptr;
    }
}

/** Helper class for SGD */
class SGD : public Optimizer {
public:
    SGD(float _lr = 1e-4, float _weight_decay = 0, const LRSchedule &_schedule = "linear") :
            Optimizer("SGD", 0, _lr, _weight_decay, _schedule) {}
};

/** Helper class for Momentum */
class Momentum : public Optimizer {
public:
    Momentum(float _lr = 1e-4, float _weight_decay = 0, float _momentum = 0.999,
             const LRSchedule &_schedule = "linear") :
            Optimizer("Momentum", 1, _lr, _weight_decay, _schedule) {
        momentum = _momentum;
    }
};

/** Helper class for AdaGrad */
class AdaGrad : public Optimizer {
public:
    AdaGrad(float _lr = 1e-4, float _weight_decay = 0, float _epsilon = 1e-10,
            const LRSchedule &_schedule = "linear") :
            Optimizer("AdaGrad", 1, _lr, _weight_decay, _schedule) {
        epsilon = _epsilon;
    }
};

/** Helper class for RMSprop */
class RMSprop : public Optimizer {
public:
    RMSprop(float _lr = 1e-4, float _weight_decay = 0, float _alpha = 0.999, float _epsilon = 1e-8,
            const LRSchedule &_schedule = "linear") :
            Optimizer("RMSprop", 1, _lr, _weight_decay, _schedule) {
        alpha = _alpha;
        epsilon = _epsilon;
    }
};

/** Helper class for Adam */
class Adam : public Optimizer {
public:
    Adam(float _lr = 1e-4, float _weight_decay = 0, float _beta1 = 0.999, float _beta2 = 0.99999, float _epsilon = 1e-8,
         const LRSchedule &_schedule = "linear") :
            Optimizer("Adam", 2, _lr, _weight_decay, _schedule) {
        beta1 = _beta1;
        beta2 = _beta2;
        epsilon = _epsilon;
    }
};

} // namespace graphvite