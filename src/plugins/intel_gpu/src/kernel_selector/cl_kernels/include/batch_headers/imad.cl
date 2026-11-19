// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

inline int imad_SW(int acc, uchar4 input, char4 weight) __attribute__((overloadable)) {
    acc += input.s0 * weight.s0;
    acc += input.s1 * weight.s1;
    acc += input.s2 * weight.s2;
    acc += input.s3 * weight.s3;
    return acc;
}

inline int imad_SW(int acc, char4 input, char4 weight) __attribute__((overloadable)) {
    acc += input.s0 * weight.s0;
    acc += input.s1 * weight.s1;
    acc += input.s2 * weight.s2;
    acc += input.s3 * weight.s3;
    return acc;
}

inline int imad_SW(int acc, char4 input, uchar4 weight) __attribute__((overloadable)) {
    acc += input.s0 * weight.s0;
    acc += input.s1 * weight.s1;
    acc += input.s2 * weight.s2;
    acc += input.s3 * weight.s3;
    return acc;
}

inline int imad_SW(int acc, uchar4 input, uchar4 weight) __attribute__((overloadable)) {
    acc += input.s0 * weight.s0;
    acc += input.s1 * weight.s1;
    acc += input.s2 * weight.s2;
    acc += input.s3 * weight.s3;
    return acc;
}


#define IMAD(_O, _I, _W) imad_SW(_O, _I, _W)
