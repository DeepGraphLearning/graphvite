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

#include "core/optimizer.h"
#include "util/gpu.cuh"
#include "util/math.h"

namespace graphvite {

/**
 * @brief TransE model
 * @tparam _Vector vector type of embeddings
 *
 * Forward: margin - L1_norm(head + relation - tail)
 * Backward: gradient of forward function
 */
template<class _Vector>
class TransE {
public:
    static const size_t dim = _Vector::dim;
    typedef _Vector Vector;
    typedef typename _Vector::Float Float;

    __host__ __device__
    static void forward(const Vector &head, const Vector &tail, const Vector &relation, Float &output, float margin) {
        output = 0;
        FOR(i, dim)
            output += abs(head[i] + relation[i] - tail[i]);
        output = margin - SUM(output);
    }

    template<OptimizerType optimizer_type>
    __host__ __device__
    static void backward(Vector &head, Vector &tail, Vector &relation,
                         float margin, Float gradient, const Optimizer &optimizer, float relation_lr_multiplier = 1,
                         Float weight = 1) {
        auto update = get_update_function<Float, optimizer_type>();
        FOR(i, dim) {
            Float h = head[i];
            Float t = tail[i];
            Float r = relation[i];
            Float s = h + r - t > 0 ? 1 : -1;
            head[i] -= (optimizer.*update)(h, -gradient * s, weight);
            tail[i] -= (optimizer.*update)(t, gradient * s, weight);
            relation[i] -= relation_lr_multiplier * (optimizer.*update)(r, -gradient * s, weight);
        }
    }

    template<OptimizerType optimizer_type>
    __host__ __device__
    static void backward(Vector &head, Vector &tail, Vector &relation,
                         Vector &head_moment1, Vector &tail_moment1, Vector &relation_moment1,
                         float margin, Float gradient, const Optimizer &optimizer, float relation_lr_multiplier = 1,
                         Float weight = 1) {
        auto update = get_update_function_1_moment<Float, optimizer_type>();
        FOR(i, dim) {
            Float h = head[i];
            Float t = tail[i];
            Float r = relation[i];
            Float s = h + r - t > 0 ? 1 : -1;
            head[i] -= (optimizer.*update)(h, -gradient * s, head_moment1[i], weight);
            tail[i] -= (optimizer.*update)(t, gradient * s, tail_moment1[i], weight);
            relation[i] -= relation_lr_multiplier * (optimizer.*update)(r, -gradient * s, relation_moment1[i], weight);
        }
    }

    template<OptimizerType optimizer_type>
    __host__ __device__
    static void backward(Vector &head, Vector &tail, Vector &relation,
                         Vector &head_moment1, Vector &tail_moment1, Vector &relation_moment1,
                         Vector &head_moment2, Vector &tail_moment2, Vector &relation_moment2,
                         float margin, Float gradient, const Optimizer &optimizer, float relation_lr_multiplier = 1,
                         Float weight = 1) {
        auto update = get_update_function_2_moment<Float, optimizer_type>();
        FOR(i, dim) {
            Float h = head[i];
            Float t = tail[i];
            Float r = relation[i];
            Float s = h + r - t > 0 ? 1 : -1;
            head[i] -= (optimizer.*update)(h, -gradient * s, head_moment1[i], head_moment2[i], weight);
            tail[i] -= (optimizer.*update)(t, gradient * s, tail_moment1[i], tail_moment2[i], weight);
            relation[i] -= relation_lr_multiplier *
                    (optimizer.*update)(r, -gradient * s, relation_moment1[i], relation_moment2[i], weight);
        }
    }
};

/**
 * @brief DistMult model
 * @tparam _Vector vector type of embeddings
 *
 * Forward: sum(head * relation * tail)
 * Backward: gradient of forward function, with l3 regularization on each parameter
 */
template<class _Vector>
class DistMult {
public:
    static const size_t dim = _Vector::dim;
    typedef _Vector Vector;
    typedef typename _Vector::Float Float;

    __host__ __device__
    static void forward(const Vector &head, const Vector &tail, const Vector &relation, Float &output,
                        float l3_regularization) {
        output = 0;
        FOR(i, dim)
            output += head[i] * relation[i] * tail[i];
        output = SUM(output);
    }

    template<OptimizerType optimizer_type>
    __host__ __device__
    static void backward(Vector &head, Vector &tail, Vector &relation,
                         float l3_regularization, Float gradient, const Optimizer &optimizer,
                         float relation_lr_multiplier = 1, Float weight = 1) {
        auto update = get_update_function<Float, optimizer_type>();
        l3_regularization *= 3;
        FOR(i, dim) {
            Float h = head[i];
            Float t = tail[i];
            Float r = relation[i];
            head[i] -= (optimizer.*update)(h, gradient * r * t + l3_regularization * abs(h) * h, weight);
            tail[i] -= (optimizer.*update)(t, gradient * h * r + l3_regularization * abs(t) * t, weight);
            relation[i] -= relation_lr_multiplier *
                    (optimizer.*update)(r, gradient * h * t + l3_regularization * abs(r) * r, weight);
        }
    }

    template<OptimizerType optimizer_type>
    __host__ __device__
    static void backward(Vector &head, Vector &tail, Vector &relation,
                         Vector &head_moment1, Vector &tail_moment1, Vector &relation_moment1,
                         float l3_regularization, Float gradient, const Optimizer &optimizer,
                         float relation_lr_multiplier = 1, Float weight = 1) {
        auto update = get_update_function_1_moment<Float, optimizer_type>();
        l3_regularization *= 3;
        FOR(i, dim) {
            Float h = head[i];
            Float t = tail[i];
            Float r = relation[i];
            head[i] -= (optimizer.*update)(h, gradient * r * t + l3_regularization * abs(h) * h,
                                           head_moment1[i], weight);
            tail[i] -= (optimizer.*update)(t, gradient * h * r + l3_regularization * abs(t) * t,
                                           tail_moment1[i], weight);
            relation[i] -= relation_lr_multiplier *
                    (optimizer.*update)(r, gradient * h * t + l3_regularization * abs(r) * r,
                                        relation_moment1[i], weight);
        }
    }

    template<OptimizerType optimizer_type>
    __host__ __device__
    static void backward(Vector &head, Vector &tail, Vector &relation,
                         Vector &head_moment1, Vector &tail_moment1, Vector &relation_moment1,
                         Vector &head_moment2, Vector &tail_moment2, Vector &relation_moment2,
                         float l3_regularization, Float gradient, const Optimizer &optimizer,
                         float relation_lr_multiplier = 1, Float weight = 1) {
        auto update = get_update_function_2_moment<Float, optimizer_type>();
        l3_regularization *= 3;
        FOR(i, dim) {
            Float h = head[i];
            Float t = tail[i];
            Float r = relation[i];
            head[i] -= (optimizer.*update)(h, gradient * r * t + l3_regularization * abs(h) * h,
                                           head_moment1[i], head_moment2[i], weight);
            tail[i] -= (optimizer.*update)(t, gradient * h * r + l3_regularization * abs(t) * t,
                                           tail_moment1[i], tail_moment2[i], weight);
            relation[i] -= relation_lr_multiplier *
                    (optimizer.*update)(r, gradient * h * t + l3_regularization * abs(r) * r,
                                        relation_moment1[i], relation_moment2[i], weight);
        }
    }
};

/**
 * @brief ComplEx model
 * @tparam _Vector vector type of embeddings
 *
 * Forward: real(sum(head * relation * conjugate(tail)))
 * Backward: gradient of forward function, with l3 regularization on each parameter
 */
template<class _Vector>
class ComplEx {
public:
    static_assert(_Vector::dim % 2 == 0, "Model `ComplEx` can only be instantiated with even-dimensional vectors");
    static const size_t dim = _Vector::dim;
    typedef _Vector Vector;
    typedef typename _Vector::Float Float;

    __host__ __device__
    static void forward(const Vector &head, const Vector &tail, const Vector &relation, Float &output,
                        float l3_regularization) {
        output = 0;
        FOR(i, dim / 2) {
            Float h_re = head[i * 2];
            Float h_im = head[i * 2 + 1];
            Float t_re = tail[i * 2];
            Float t_im = tail[i * 2 + 1];
            Float r_re = relation[i * 2];
            Float r_im = relation[i * 2 + 1];
            Float product_re = h_re * r_re - h_im * r_im;
            Float product_im = h_re * r_im + h_im * r_re;
            output += product_re * t_re + product_im * t_im;
        }
        output = SUM(output);
    }

    template<OptimizerType optimizer_type>
    __host__ __device__
    static void backward(Vector &head, Vector &tail, Vector &relation,
                         float l3_regularization, Float gradient, const Optimizer &optimizer,
                         float relation_lr_multiplier = 1, Float weight = 1) {
        auto update = get_update_function<Float, optimizer_type>();
        l3_regularization *= 3;
        FOR(i, dim / 2) {
            Float h_re = head[i * 2];
            Float h_im = head[i * 2 + 1];
            Float t_re = tail[i * 2];
            Float t_im = tail[i * 2 + 1];
            Float r_re = relation[i * 2];
            Float r_im = relation[i * 2 + 1];
            // head
            Float h_re_grad = gradient * (r_re * t_re + r_im * t_im);
            Float h_im_grad = gradient * (-r_im * t_re + r_re * t_im);
            head[i * 2] -= (optimizer.*update)(h_re, h_re_grad + l3_regularization * abs(h_re) * h_re, weight);
            head[i * 2 + 1] -= (optimizer.*update)(h_im, h_im_grad + l3_regularization * abs(h_im) * h_im, weight);
            // tail
            Float t_re_grad = gradient * (h_re * r_re - h_im * r_im);
            Float t_im_grad = gradient * (h_re * r_im + h_im * r_re);
            tail[i * 2] -= (optimizer.*update)(t_re, t_re_grad + l3_regularization * abs(t_re) * t_re, weight);
            tail[i * 2 + 1] -= (optimizer.*update)(t_im, t_im_grad + l3_regularization * abs(t_im) * t_im, weight);
            // relation
            Float r_re_grad = gradient * (h_re * t_re + h_im * t_im);
            Float r_im_grad = gradient * (-h_im * t_re + h_re * t_im);
            relation[i * 2] -= relation_lr_multiplier *
                    (optimizer.*update)(r_re, r_re_grad + l3_regularization * abs(r_re) * r_re, weight);
            relation[i * 2 + 1] -= relation_lr_multiplier *
                    (optimizer.*update)(r_im, r_im_grad + l3_regularization * abs(r_im) * r_im, weight);
        }
    }

    template<OptimizerType optimizer_type>
    __host__ __device__
    static void backward(Vector &head, Vector &tail, Vector &relation,
                         Vector &head_moment1, Vector &tail_moment1, Vector &relation_moment1,
                         float l3_regularization, Float gradient, const Optimizer &optimizer,
                         float relation_lr_multiplier = 1, Float weight = 1) {
        auto update = get_update_function_1_moment<Float, optimizer_type>();
        l3_regularization *= 3;
        FOR(i, dim / 2) {
            Float h_re = head[i * 2];
            Float h_im = head[i * 2 + 1];
            Float t_re = tail[i * 2];
            Float t_im = tail[i * 2 + 1];
            Float r_re = relation[i * 2];
            Float r_im = relation[i * 2 + 1];
            // head
            Float h_re_grad = gradient * (r_re * t_re + r_im * t_im);
            Float h_im_grad = gradient * (-r_im * t_re + r_re * t_im);
            head[i * 2] -= (optimizer.*update)(h_re, h_re_grad + l3_regularization * abs(h_re) * h_re,
                                               head_moment1[i * 2], weight);
            head[i * 2 + 1] -= (optimizer.*update)(h_im, h_im_grad + l3_regularization * abs(h_im) * h_im,
                                                   head_moment1[i * 2 + 1], weight);
            // tail
            Float t_re_grad = gradient * (h_re * r_re - h_im * r_im);
            Float t_im_grad = gradient * (h_re * r_im + h_im * r_re);
            tail[i * 2] -= (optimizer.*update)(t_re, t_re_grad + l3_regularization * abs(t_re) * t_re,
                                               tail_moment1[i * 2], weight);
            tail[i * 2 + 1] -= (optimizer.*update)(t_im, t_im_grad + l3_regularization * abs(t_im) * t_im,
                                                   tail_moment1[i * 2 + 1], weight);
            // relation
            Float r_re_grad = gradient * (h_re * t_re + h_im * t_im);
            Float r_im_grad = gradient * (-h_im * t_re + h_re * t_im);
            relation[i * 2] -= relation_lr_multiplier *
                    (optimizer.*update)(r_re, r_re_grad + l3_regularization * abs(r_re) * r_re,
                                        relation_moment1[i], weight);
            relation[i * 2 + 1] -= relation_lr_multiplier *
                    (optimizer.*update)(r_im, r_im_grad + l3_regularization * abs(r_im) * r_im,
                                        relation_moment1[i], weight);
        }
    }

    template<OptimizerType optimizer_type>
    __host__ __device__
    static void backward(Vector &head, Vector &tail, Vector &relation,
                         Vector &head_moment1, Vector &tail_moment1, Vector &relation_moment1,
                         Vector &head_moment2, Vector &tail_moment2, Vector &relation_moment2,
                         float l3_regularization, Float gradient, const Optimizer &optimizer,
                         float relation_lr_multiplier = 1, Float weight = 1) {
        auto update = get_update_function_2_moment<Float, optimizer_type>();
        l3_regularization *= 3;
        FOR(i, dim / 2) {
            Float h_re = head[i * 2];
            Float h_im = head[i * 2 + 1];
            Float t_re = tail[i * 2];
            Float t_im = tail[i * 2 + 1];
            Float r_re = relation[i * 2];
            Float r_im = relation[i * 2 + 1];
            // head
            Float h_re_grad = gradient * (r_re * t_re + r_im * t_im);
            Float h_im_grad = gradient * (-r_im * t_re + r_re * t_im);
            head[i * 2] -= (optimizer.*update)(h_re, h_re_grad + l3_regularization * abs(h_re) * h_re,
                                               head_moment1[i * 2], head_moment2[i * 2], weight);
            head[i * 2 + 1] -= (optimizer.*update)(h_im, h_im_grad + l3_regularization * abs(h_im) * h_im,
                                                   head_moment1[i * 2 + 1], head_moment2[i * 2 + 1], weight);
            // tail
            Float t_re_grad = gradient * (h_re * r_re - h_im * r_im);
            Float t_im_grad = gradient * (h_re * r_im + h_im * r_re);
            tail[i * 2] -= (optimizer.*update)(t_re, t_re_grad + l3_regularization * abs(t_re) * t_re,
                                               tail_moment1[i * 2], tail_moment2[i * 2], weight);
            tail[i * 2 + 1] -= (optimizer.*update)(t_im, t_im_grad + l3_regularization * abs(t_im) * t_im,
                                                   tail_moment1[i * 2 + 1], tail_moment2[i * 2 + 1], weight);
            // relation
            Float r_re_grad = gradient * (h_re * t_re + h_im * t_im);
            Float r_im_grad = gradient * (-h_im * t_re + h_re * t_im);
            relation[i * 2] -= relation_lr_multiplier *
                    (optimizer.*update)(r_re, r_re_grad + l3_regularization * abs(r_re) * r_re,
                                        relation_moment1[i], relation_moment2[i], weight);
            relation[i * 2 + 1] -= relation_lr_multiplier *
                    (optimizer.*update)(r_im, r_im_grad + l3_regularization * abs(r_im) * r_im,
                                        relation_moment1[i], relation_moment2[i], weight);
        }
    }
};

/**
 * @brief SimplE model
 * @tparam _Vector vector type of embeddings
 *
 * Forward: sum(head * relation * flip(tail))
 * Backward: gradient of forward function, with l3 regularization on each parameter
 */
template<class _Vector>
class SimplE {
public:
    static_assert(_Vector::dim % 2 == 0, "Model `SimplE` can only be instantiated with even-dimensional vectors");
    static const size_t dim = _Vector::dim;
    typedef _Vector Vector;
    typedef typename _Vector::Float Float;

    __host__ __device__
    static void forward(const Vector &head, const Vector &tail, const Vector &relation, Float &output,
                        float l3_regularization) {
        output = 0;
        FOR(i, dim) {
            int j = i ^ 1;
            output += head[i] * relation[i] * tail[j];
        }
        output = SUM(output);
    }

    template<OptimizerType optimizer_type>
    __host__ __device__
    static void backward(Vector &head, Vector &tail, Vector &relation,
                         float l3_regularization, Float gradient, const Optimizer &optimizer,
                         float relation_lr_multiplier = 1, Float weight = 1) {
        auto update = get_update_function<Float, optimizer_type>();
        l3_regularization *= 3;
        FOR(i, dim) {
            int j = i ^ 1;
            Float h = head[i];
            Float t = tail[j];
            Float r = relation[i];
            head[i] -= (optimizer.*update)(h, gradient * r * t + l3_regularization * abs(h) * h, weight);
            tail[j] -= (optimizer.*update)(t, gradient * h * r + l3_regularization * abs(t) * t, weight);
            relation[i] -= relation_lr_multiplier *
                    (optimizer.*update)(r, gradient * h * t + l3_regularization * abs(r) * r, weight);
        }
    }

    template<OptimizerType optimizer_type>
    __host__ __device__
    static void backward(Vector &head, Vector &tail, Vector &relation,
                         Vector &head_moment1, Vector &tail_moment1, Vector &relation_moment1,
                         float l3_regularization, Float gradient, const Optimizer &optimizer,
                         float relation_lr_multiplier = 1, Float weight = 1) {
        auto update = get_update_function_1_moment<Float, optimizer_type>();
        l3_regularization *= 3;
        FOR(i, dim) {
            int j = i ^ 1;
            Float h = head[i];
            Float t = tail[j];
            Float r = relation[i];
            head[i] -= (optimizer.*update)(h, gradient * r * t + l3_regularization * abs(h) * h,
                                           head_moment1[i], weight);
            tail[j] -= (optimizer.*update)(t, gradient * h * r + l3_regularization * abs(t) * t,
                                           tail_moment1[j], weight);
            relation[i] -= relation_lr_multiplier *
                    (optimizer.*update)(r, gradient * h * t + l3_regularization * abs(r) * r,
                                        relation_moment1[i], weight);
        }
    }

    template<OptimizerType optimizer_type>
    __host__ __device__
    static void backward(Vector &head, Vector &tail, Vector &relation,
                         Vector &head_moment1, Vector &tail_moment1, Vector &relation_moment1,
                         Vector &head_moment2, Vector &tail_moment2, Vector &relation_moment2,
                         float l3_regularization, Float gradient, const Optimizer &optimizer,
                         float relation_lr_multiplier = 1, Float weight = 1) {
        auto update = get_update_function_2_moment<Float, optimizer_type>();
        l3_regularization *= 3;
        FOR(i, dim) {
            int j = i ^ 1;
            Float h = head[i];
            Float t = tail[j];
            Float r = relation[i];
            head[i] -= (optimizer.*update)(h, gradient * r * t + l3_regularization * abs(h) * h,
                                           head_moment1[i], head_moment2[i], weight);
            tail[j] -= (optimizer.*update)(t, gradient * h * r + l3_regularization * abs(t) * t,
                                           tail_moment1[j], tail_moment2[j], weight);
            relation[i] -= relation_lr_multiplier *
                    (optimizer.*update)(r, gradient * h * t + l3_regularization * abs(r) * r,
                                        relation_moment1[i], relation_moment2[i], weight);
        }
    }
};

/**
 * @brief RotatE model
 * @tparam _Vector vector type of embeddings
 *
 * Forward: margin - L1_norm(head * relation - tail), with constraint L2_norm(relation[*]) = 1
 * Backward: gradient of forward function
 *
 * In practice, the relation is reparameterized as a phase vector to remove the constraint.
 */
template<class _Vector>
class RotatE {
public:
    static_assert(_Vector::dim % 2 == 0, "Model `RotatE` can only be instantiated with even-dimensional vectors");
    static const size_t dim = _Vector::dim;
    typedef _Vector Vector;
    typedef typename _Vector::Float Float;

    __host__ __device__
    static void forward(const Vector &head, const Vector &tail, const Vector &relation, Float &output, float margin) {
        output = 0;
        FOR(i, dim / 2) {
            Float h_re = head[i * 2];
            Float h_im = head[i * 2 + 1];
            Float t_re = tail[i * 2];
            Float t_im = tail[i * 2 + 1];
            Float phase = relation[i];
            Float r_re = cos(phase);
            Float r_im = sin(phase);
            Float distance_re = h_re * r_re - h_im * r_im - t_re;
            Float distance_im = h_re * r_im + h_im * r_re - t_im;
            output += sqrt(distance_re * distance_re + distance_im * distance_im);
        }
        output = margin - SUM(output);
    }

    template<OptimizerType optimizer_type>
    __host__ __device__
    static void backward(Vector &head, Vector &tail, Vector &relation,
                         float margin, Float gradient, const Optimizer &optimizer,
                         float relation_lr_multiplier = 1, Float weight = 1) {
        auto update = get_update_function<Float, optimizer_type>();
        FOR(i, dim / 2) {
            Float phase = relation[i];
            Float r_re = cos(phase);
            Float r_im = sin(phase);
            Float h_re = head[i * 2];
            Float h_im = head[i * 2 + 1];
            Float t_re = tail[i * 2];
            Float t_im = tail[i * 2 + 1];
            Float distance_re = h_re * r_re - h_im * r_im - t_re;
            Float distance_im = h_re * r_im + h_im * r_re - t_im;
            Float grad = gradient / (sqrt(distance_re * distance_re + distance_im * distance_im) + kEpsilon);
            // head
            Float head_re_grad = -grad * (distance_re * r_re + distance_im * r_im);
            Float head_im_grad = -grad * (-distance_re * r_im + distance_im * r_re);
            head[i * 2] -= (optimizer.*update)(h_re, head_re_grad, weight);
            head[i * 2 + 1] -= (optimizer.*update)(h_im, head_im_grad, weight);
            // tail
            tail[i * 2] -= (optimizer.*update)(t_re, grad * distance_re, weight);
            tail[i * 2 + 1] -= (optimizer.*update)(t_im, grad * distance_im, weight);
            // relation
            Float relation_grad =
                    -grad * (distance_re * (h_re * -r_im + h_im * -r_re) + distance_im * (h_re * r_re + h_im * -r_im));
            relation[i] -= relation_lr_multiplier * (optimizer.*update)(phase, relation_grad, weight);
        }
    }

    template<OptimizerType optimizer_type>
    __host__ __device__
    static void backward(Vector &head, Vector &tail, Vector &relation,
                         Vector &head_moment1, Vector &tail_moment1, Vector &relation_moment1,
                         float margin, Float gradient, const Optimizer &optimizer,
                         float relation_lr_multiplier = 1, Float weight = 1) {
        auto update = get_update_function_1_moment<Float, optimizer_type>();
        FOR(i, dim / 2) {
            Float phase = relation[i];
            Float r_re = cos(phase);
            Float r_im = sin(phase);
            Float h_re = head[i * 2];
            Float h_im = head[i * 2 + 1];
            Float t_re = tail[i * 2];
            Float t_im = tail[i * 2 + 1];
            Float distance_re = h_re * r_re - h_im * r_im - t_re;
            Float distance_im = h_re * r_im + h_im * r_re - t_im;
            Float grad = gradient / (sqrt(distance_re * distance_re + distance_im * distance_im) + kEpsilon);
            // head
            Float head_re_grad = -grad * (distance_re * r_re + distance_im * r_im);
            Float head_im_grad = -grad * (-distance_re * r_im + distance_im * r_re);
            head[i * 2] -= (optimizer.*update)(h_re, head_re_grad, head_moment1[i * 2], weight);
            head[i * 2 + 1] -= (optimizer.*update)(h_im, head_im_grad, head_moment1[i * 2 + 1], weight);
            // tail
            tail[i * 2] -= (optimizer.*update)(t_re, grad * distance_re, tail_moment1[i * 2], weight);
            tail[i * 2 + 1] -= (optimizer.*update)(t_im, grad * distance_im, tail_moment1[i * 2 + 1], weight);
            // relation
            Float relation_grad =
                    -grad * (distance_re * (h_re * -r_im + h_im * -r_re) + distance_im * (h_re * r_re + h_im * -r_im));
            relation[i] -= relation_lr_multiplier *
                    (optimizer.*update)(phase, relation_grad, relation_moment1[i], weight);
        }
    }

    template<OptimizerType optimizer_type>
    __host__ __device__
    static void backward(Vector &head, Vector &tail, Vector &relation,
                         Vector &head_moment1, Vector &tail_moment1, Vector &relation_moment1,
                         Vector &head_moment2, Vector &tail_moment2, Vector &relation_moment2,
                         float margin, Float gradient, const Optimizer &optimizer,
                         float relation_lr_multiplier = 1, Float weight = 1) {
        auto update = get_update_function_2_moment<Float, optimizer_type>();
        FOR(i, dim / 2) {
            Float phase = relation[i];
            Float r_re = cos(phase);
            Float r_im = sin(phase);
            Float h_re = head[i * 2];
            Float h_im = head[i * 2 + 1];
            Float t_re = tail[i * 2];
            Float t_im = tail[i * 2 + 1];
            Float distance_re = h_re * r_re - h_im * r_im - t_re;
            Float distance_im = h_re * r_im + h_im * r_re - t_im;
            Float grad = gradient / (sqrt(distance_re * distance_re + distance_im * distance_im) + kEpsilon);
            // head
            Float head_re_grad = -grad * (distance_re * r_re + distance_im * r_im);
            Float head_im_grad = -grad * (-distance_re * r_im + distance_im * r_re);
            head[i * 2] -= (optimizer.*update)(h_re, head_re_grad,
                                               head_moment1[i * 2], head_moment2[i * 2], weight);
            head[i * 2 + 1] -= (optimizer.*update)(h_im, head_im_grad,
                                                   head_moment1[i * 2 + 1], head_moment2[i * 2 + 1], weight);
            // tail
            tail[i * 2] -= (optimizer.*update)(t_re, grad * distance_re,
                                               tail_moment1[i * 2], tail_moment2[i * 2], weight);
            tail[i * 2 + 1] -= (optimizer.*update)(t_im, grad * distance_im,
                                                   tail_moment1[i * 2 + 1], tail_moment2[i * 2 + 1], weight);
            // relation
            Float relation_grad =
                    -grad * (distance_re * (h_re * -r_im + h_im * -r_re) + distance_im * (h_re * r_re + h_im * -r_im));
            relation[i] -= relation_lr_multiplier *
                    (optimizer.*update)(phase, relation_grad, relation_moment1[i], relation_moment2[i], weight);
        }
    }
};

/**
 * @brief QuatE model
 * @tparam _Vector vector type of embeddings
 *
 * Forward: sum(hamilton_product(head, relation) * tail), with constraint L2_norm(relation[*]) = 1
 * Backward: gradient of forward function
 */
template<class _Vector>
class QuatE {
public:
    static_assert(_Vector::dim % 4 == 0,
                  "Model `QuatE` can only be instantiated with vector dimensions divisible by 4");
    static const size_t dim = _Vector::dim;
    typedef _Vector Vector;
    typedef typename _Vector::Float Float;

    __host__ __device__
    static void forward(const Vector &head, const Vector &tail, const Vector &relation, Float &output,
                        float l3_regularization) {
        output = 0;
        FOR(i, dim / 4) {
            Float h_r = head[i * 4];
            Float h_i = head[i * 4 + 1];
            Float h_j = head[i * 4 + 2];
            Float h_k = head[i * 4 + 3];
            Float r_r = relation[i * 4];
            Float r_i = relation[i * 4 + 1];
            Float r_j = relation[i * 4 + 2];
            Float r_k = relation[i * 4 + 3];
            Float t_r = tail[i * 4];
            Float t_i = tail[i * 4 + 1];
            Float t_j = tail[i * 4 + 2];
            Float t_k = tail[i * 4 + 3];
            Float r_norm = sqrt(r_r * r_r + r_i * r_i + r_j * r_j + r_k * r_k);
            Float product_r = h_r * r_r - h_i * r_i - h_j * r_j - h_k * r_k;
            Float product_i = h_r * r_i + h_i * r_r + h_j * r_k - h_k * r_j;
            Float product_j = h_r * r_j - h_i * r_k + h_j * r_r + h_k * r_i;
            Float product_k = h_r * r_k + h_i * r_j - h_j * r_i + h_k * r_r;
            output += (product_r * t_r + product_i * t_i + product_j * t_j + product_k * t_k) / (r_norm + kEpsilon);
        }
        output = SUM(output);
    }

    template<OptimizerType optimizer_type>
    __host__ __device__
    static void backward(Vector &head, Vector &tail, Vector &relation,
                         float l3_regularization, Float gradient, const Optimizer &optimizer,
                         float relation_lr_multiplier = 1, Float weight = 1) {
        auto update = get_update_function<Float, optimizer_type>();
        l3_regularization *= 3;
        FOR(i, dim / 4) {
            Float h_r = head[i * 4];
            Float h_i = head[i * 4 + 1];
            Float h_j = head[i * 4 + 2];
            Float h_k = head[i * 4 + 3];
            Float r_r = relation[i * 4];
            Float r_i = relation[i * 4 + 1];
            Float r_j = relation[i * 4 + 2];
            Float r_k = relation[i * 4 + 3];
            Float t_r = tail[i * 4];
            Float t_i = tail[i * 4 + 1];
            Float t_j = tail[i * 4 + 2];
            Float t_k = tail[i * 4 + 3];
            Float r_norm = sqrt(r_r * r_r + r_i * r_i + r_j * r_j + r_k * r_k);
            Float grad = gradient / (r_norm + kEpsilon);
            // head
            Float h_r_grad = grad * (r_r * t_r + r_i * t_i + r_j * t_j + r_k * t_k);
            Float h_i_grad = grad * (-r_i * t_r + r_r * t_i - r_k * t_j + r_j * t_k);
            Float h_j_grad = grad * (-r_j * t_r + r_k * t_i + r_r * t_j - r_i * t_k);
            Float h_k_grad = grad * (-r_k * t_r - r_j * t_i + r_i * t_j + r_r * t_k);
            head[i * 4] -= (optimizer.*update)(h_r, h_r_grad + l3_regularization * abs(h_r) * h_r, weight);
            head[i * 4 + 1] -= (optimizer.*update)(h_i, h_i_grad + l3_regularization * abs(h_i) * h_i, weight);
            head[i * 4 + 2] -= (optimizer.*update)(h_j, h_j_grad + l3_regularization * abs(h_j) * h_j, weight);
            head[i * 4 + 3] -= (optimizer.*update)(h_k, h_k_grad + l3_regularization * abs(h_k) * h_k, weight);
            // tail
            Float t_r_grad = grad * (h_r * r_r - h_i * r_i - h_j * r_j - h_k * r_k);
            Float t_i_grad = grad * (h_r * r_i + h_i * r_r + h_j * r_k - h_k * r_j);
            Float t_j_grad = grad * (h_r * r_j - h_i * r_k + h_j * r_r + h_k * r_i);
            Float t_k_grad = grad * (h_r * r_k + h_i * r_j - h_j * r_i + h_k * r_r);
            tail[i * 4] -= (optimizer.*update)(t_r, t_r_grad + l3_regularization * abs(t_r) * t_r, weight);
            tail[i * 4 + 1] -= (optimizer.*update)(t_i, t_i_grad + l3_regularization * abs(t_i) * t_i, weight);
            tail[i * 4 + 2] -= (optimizer.*update)(t_j, t_j_grad + l3_regularization * abs(t_j) * t_j, weight);
            tail[i * 4 + 3] -= (optimizer.*update)(t_k, t_k_grad + l3_regularization * abs(t_k) * t_k, weight);
            // relation
            Float r_r_grad = grad * (h_r * t_r + h_i * t_i + h_j * t_j + h_k * t_k);
            Float r_i_grad = grad * (-h_i * t_r + h_r * t_i + h_k * t_j - h_j * t_k);
            Float r_j_grad = grad * (-h_j * t_r - h_k * t_i + h_r * t_j + h_i * t_k);
            Float r_k_grad = grad * (-h_k * t_r + h_j * t_i - h_i * t_j + h_r * t_k);
            relation[i * 4] -= relation_lr_multiplier *
                               (optimizer.*update)(r_r, r_r_grad + l3_regularization * abs(r_r) * r_r, weight);
            relation[i * 4 + 1] -= relation_lr_multiplier *
                                   (optimizer.*update)(r_i, r_i_grad + l3_regularization * abs(r_i) * r_i, weight);
            relation[i * 4 + 2] -= relation_lr_multiplier *
                                   (optimizer.*update)(r_j, r_j_grad + l3_regularization * abs(r_j) * r_j, weight);
            relation[i * 4 + 3] -= relation_lr_multiplier *
                                   (optimizer.*update)(r_k, r_k_grad + l3_regularization * abs(r_k) * r_k, weight);
        }
    }

    template<OptimizerType optimizer_type>
    __host__ __device__
    static void backward(Vector &head, Vector &tail, Vector &relation,
                         Vector &head_moment1, Vector &tail_moment1, Vector &relation_moment1,
                         float l3_regularization, Float gradient, const Optimizer &optimizer,
                         float relation_lr_multiplier = 1, Float weight = 1) {
        auto update = get_update_function_1_moment<Float, optimizer_type>();
        l3_regularization *= 3;
        FOR(i, dim / 4) {
            Float h_r = head[i * 4];
            Float h_i = head[i * 4 + 1];
            Float h_j = head[i * 4 + 2];
            Float h_k = head[i * 4 + 3];
            Float r_r = relation[i * 4];
            Float r_i = relation[i * 4 + 1];
            Float r_j = relation[i * 4 + 2];
            Float r_k = relation[i * 4 + 3];
            Float t_r = tail[i * 4];
            Float t_i = tail[i * 4 + 1];
            Float t_j = tail[i * 4 + 2];
            Float t_k = tail[i * 4 + 3];
            Float r_norm = sqrt(r_r * r_r + r_i * r_i + r_j * r_j + r_k * r_k);
            Float grad = gradient / (r_norm + kEpsilon);
            // head
            Float h_r_grad = grad * (r_r * t_r + r_i * t_i + r_j * t_j + r_k * t_k);
            Float h_i_grad = grad * (-r_i * t_r + r_r * t_i - r_k * t_j + r_j * t_k);
            Float h_j_grad = grad * (-r_j * t_r + r_k * t_i + r_r * t_j - r_i * t_k);
            Float h_k_grad = grad * (-r_k * t_r - r_j * t_i + r_i * t_j + r_r * t_k);
            head[i * 4] -= (optimizer.*update)(h_r, h_r_grad + l3_regularization * abs(h_r) * h_r,
                                               head_moment1[i * 4], weight);
            head[i * 4 + 1] -= (optimizer.*update)(h_i, h_i_grad + l3_regularization * abs(h_i) * h_i,
                                                   head_moment1[i * 4 + 1], weight);
            head[i * 4 + 2] -= (optimizer.*update)(h_j, h_j_grad + l3_regularization * abs(h_j) * h_j,
                                                   head_moment1[i * 4 + 2], weight);
            head[i * 4 + 3] -= (optimizer.*update)(h_k, h_k_grad + l3_regularization * abs(h_k) * h_k,
                                                   head_moment1[i * 4 + 3], weight);
            // tail
            Float t_r_grad = grad * (h_r * r_r - h_i * r_i - h_j * r_j - h_k * r_k);
            Float t_i_grad = grad * (h_r * r_i + h_i * r_r + h_j * r_k - h_k * r_j);
            Float t_j_grad = grad * (h_r * r_j - h_i * r_k + h_j * r_r + h_k * r_i);
            Float t_k_grad = grad * (h_r * r_k + h_i * r_j - h_j * r_i + h_k * r_r);
            tail[i * 4] -= (optimizer.*update)(t_r, t_r_grad + l3_regularization * abs(t_r) * t_r,
                                               tail_moment1[i * 4], weight);
            tail[i * 4 + 1] -= (optimizer.*update)(t_i, t_i_grad + l3_regularization * abs(t_i) * t_i,
                                                   tail_moment1[i * 4 + 1], weight);
            tail[i * 4 + 2] -= (optimizer.*update)(t_j, t_j_grad + l3_regularization * abs(t_j) * t_j,
                                                   tail_moment1[i * 4 + 2], weight);
            tail[i * 4 + 3] -= (optimizer.*update)(t_k, t_k_grad + l3_regularization * abs(t_k) * t_k,
                                                   tail_moment1[i * 4 + 3], weight);
            // relation
            Float r_r_grad = grad * (h_r * t_r + h_i * t_i + h_j * t_j + h_k * t_k);
            Float r_i_grad = grad * (-h_i * t_r + h_r * t_i + h_k * t_j - h_j * t_k);
            Float r_j_grad = grad * (-h_j * t_r - h_k * t_i + h_r * t_j + h_i * t_k);
            Float r_k_grad = grad * (-h_k * t_r + h_j * t_i - h_i * t_j + h_r * t_k);
            relation[i * 4] -= relation_lr_multiplier *
                               (optimizer.*update)(r_r, r_r_grad + l3_regularization * abs(r_r) * r_r,
                                                   relation_moment1[i * 4], weight);
            relation[i * 4 + 1] -= relation_lr_multiplier *
                                   (optimizer.*update)(r_i, r_i_grad + l3_regularization * abs(r_i) * r_i,
                                                       relation_moment1[i * 4 + 1], weight);
            relation[i * 4 + 2] -= relation_lr_multiplier *
                                   (optimizer.*update)(r_j, r_j_grad + l3_regularization * abs(r_j) * r_j,
                                                       relation_moment1[i * 4 + 2], weight);
            relation[i * 4 + 3] -= relation_lr_multiplier *
                                   (optimizer.*update)(r_k, r_k_grad + l3_regularization * abs(r_k) * r_k,
                                                       relation_moment1[i * 4 + 3], weight);
        }
    }

    template<OptimizerType optimizer_type>
    __host__ __device__
    static void backward(Vector &head, Vector &tail, Vector &relation,
                         Vector &head_moment1, Vector &tail_moment1, Vector &relation_moment1,
                         Vector &head_moment2, Vector &tail_moment2, Vector &relation_moment2,
                         float l3_regularization, Float gradient, const Optimizer &optimizer,
                         float relation_lr_multiplier = 1, Float weight = 1) {
        auto update = get_update_function_2_moment<Float, optimizer_type>();
        l3_regularization *= 3;
        FOR(i, dim / 4) {
            Float h_r = head[i * 4];
            Float h_i = head[i * 4 + 1];
            Float h_j = head[i * 4 + 2];
            Float h_k = head[i * 4 + 3];
            Float r_r = relation[i * 4];
            Float r_i = relation[i * 4 + 1];
            Float r_j = relation[i * 4 + 2];
            Float r_k = relation[i * 4 + 3];
            Float t_r = tail[i * 4];
            Float t_i = tail[i * 4 + 1];
            Float t_j = tail[i * 4 + 2];
            Float t_k = tail[i * 4 + 3];
            Float r_norm = sqrt(r_r * r_r + r_i * r_i + r_j * r_j + r_k * r_k);
            Float grad = gradient / (r_norm + kEpsilon);
            // head
            Float h_r_grad = grad * (r_r * t_r + r_i * t_i + r_j * t_j + r_k * t_k);
            Float h_i_grad = grad * (-r_i * t_r + r_r * t_i - r_k * t_j + r_j * t_k);
            Float h_j_grad = grad * (-r_j * t_r + r_k * t_i + r_r * t_j - r_i * t_k);
            Float h_k_grad = grad * (-r_k * t_r - r_j * t_i + r_i * t_j + r_r * t_k);
            head[i * 4] -= (optimizer.*update)(h_r, h_r_grad + l3_regularization * abs(h_r) * h_r,
                                               head_moment1[i * 4], head_moment2[i * 4], weight);
            head[i * 4 + 1] -= (optimizer.*update)(h_i, h_i_grad + l3_regularization * abs(h_i) * h_i,
                                                   head_moment1[i * 4 + 1], head_moment2[i * 4 + 1], weight);
            head[i * 4 + 2] -= (optimizer.*update)(h_j, h_j_grad + l3_regularization * abs(h_j) * h_j,
                                                   head_moment1[i * 4 + 2], head_moment2[i * 4 + 2], weight);
            head[i * 4 + 3] -= (optimizer.*update)(h_k, h_k_grad + l3_regularization * abs(h_k) * h_k,
                                                   head_moment1[i * 4 + 3], head_moment2[i * 4 + 3], weight);
            // tail
            Float t_r_grad = grad * (h_r * r_r - h_i * r_i - h_j * r_j - h_k * r_k);
            Float t_i_grad = grad * (h_r * r_i + h_i * r_r + h_j * r_k - h_k * r_j);
            Float t_j_grad = grad * (h_r * r_j - h_i * r_k + h_j * r_r + h_k * r_i);
            Float t_k_grad = grad * (h_r * r_k + h_i * r_j - h_j * r_i + h_k * r_r);
            tail[i * 4] -= (optimizer.*update)(t_r, t_r_grad + l3_regularization * abs(t_r) * t_r,
                                               tail_moment1[i * 4], tail_moment2[i * 4], weight);
            tail[i * 4 + 1] -= (optimizer.*update)(t_i, t_i_grad + l3_regularization * abs(t_i) * t_i,
                                                   tail_moment1[i * 4 + 1], tail_moment2[i * 4 + 1], weight);
            tail[i * 4 + 2] -= (optimizer.*update)(t_j, t_j_grad + l3_regularization * abs(t_j) * t_j,
                                                   tail_moment1[i * 4 + 2], tail_moment2[i * 4 + 2], weight);
            tail[i * 4 + 3] -= (optimizer.*update)(t_k, t_k_grad + l3_regularization * abs(t_k) * t_k,
                                                   tail_moment1[i * 4 + 3], tail_moment2[i * 4 + 3], weight);
            // relation
            Float r_r_grad = grad * (h_r * t_r + h_i * t_i + h_j * t_j + h_k * t_k);
            Float r_i_grad = grad * (-h_i * t_r + h_r * t_i + h_k * t_j - h_j * t_k);
            Float r_j_grad = grad * (-h_j * t_r - h_k * t_i + h_r * t_j + h_i * t_k);
            Float r_k_grad = grad * (-h_k * t_r + h_j * t_i - h_i * t_j + h_r * t_k);
            relation[i * 4] -= relation_lr_multiplier *
                               (optimizer.*update)(r_r, r_r_grad + l3_regularization * abs(r_r) * r_r,
                                                   relation_moment1[i * 4], relation_moment2[i * 4], weight);
            relation[i * 4 + 1] -= relation_lr_multiplier *
                                   (optimizer.*update)(r_i, r_i_grad + l3_regularization * abs(r_i) * r_i,
                                                       relation_moment1[i * 4 + 1], relation_moment2[i * 4 + 1], weight);
            relation[i * 4 + 2] -= relation_lr_multiplier *
                                   (optimizer.*update)(r_j, r_j_grad + l3_regularization * abs(r_j) * r_j,
                                                       relation_moment1[i * 4 + 2], relation_moment2[i * 4 + 2], weight);
            relation[i * 4 + 3] -= relation_lr_multiplier *
                                   (optimizer.*update)(r_k, r_k_grad + l3_regularization * abs(r_k) * r_k,
                                                       relation_moment1[i * 4 + 3], relation_moment2[i * 4 + 3], weight);
        }
    }
};

}