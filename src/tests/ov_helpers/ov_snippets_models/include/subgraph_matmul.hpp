// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "./snippets_helpers.hpp"
#include "snippets/utils/utils.hpp"

/* This file contains definitions of relatively simple functions (models) that will be used
 * to test snippets-specific behavior. All the functions are expected to be direct descendants of
 * SnippetsFunctionBase, so their constructors take only one (inputShapes) argument.
 */

namespace ov {
namespace test {
namespace snippets {
enum class MatMulType { MatMul, FullyConnected };
std::ostream &operator<<(std::ostream& os, MatMulType type);

class MatMulFunctionBase : public SnippetsFunctionBase {
public:
    explicit MatMulFunctionBase(const std::vector<PartialShape>& inputShapes,
                                MatMulType type,
                                const std::vector<ov::element::Type>& precisions = {});

    virtual std::set<size_t> get_constant_input_idces() const {
        return matmul_type == MatMulType::FullyConnected ? std::set<size_t>{1} : std::set<size_t>{};
    }

protected:
    void validate_function(const std::shared_ptr<Model> &f) const override;

    std::vector<ov::element::Type> precisions;
    MatMulType matmul_type;
};

/// Minimal graph to test MatMul support
/// Tokenized simply by starting subgraph,
//   in1        in2
//        Matmul
//         Result
class MatMulFunction : public MatMulFunctionBase {
public:
    explicit MatMulFunction(const std::vector<PartialShape>& inputShapes,
                            const std::vector<ov::element::Type>& precisions,
                            MatMulType type,
                            bool transpose_b = false)
        : MatMulFunctionBase(inputShapes, type, precisions), transpose_b(transpose_b) {}

protected:
    std::shared_ptr<ov::Model> initOriginal() const override;
    std::shared_ptr<ov::Model> initReference() const override;

    bool transpose_b;
};

class FQMatMulFunction : public MatMulFunctionBase {
public:
    explicit FQMatMulFunction(const std::vector<PartialShape>& inputShapes, MatMulType type, int pos = -1)
        : MatMulFunctionBase(inputShapes, type), pos(pos) {
        OPENVINO_ASSERT(inputShapes.size() == 2, "Got invalid number of input shapes");
        OPENVINO_ASSERT(pos >=-1 && pos <= 2, "Got invalid transpose position");
        if (type == MatMulType::FullyConnected)
            OPENVINO_ASSERT(pos != 1, "transpose on B input is not supported for FullyConnected matmul type");
    }

protected:
    std::shared_ptr<ov::Model> initOriginal() const override;

    int pos = -1;
};

// As same as MatMulFunction but with biases
class MatMulBiasFunction : public MatMulFunctionBase {
public:
    explicit MatMulBiasFunction(const std::vector<PartialShape>& inputShapes,
                                const std::vector<ov::element::Type>& precisions,
                                MatMulType type)
        : MatMulFunctionBase(inputShapes, type, precisions) {
        OPENVINO_ASSERT(input_shapes.size() == 3, "Got invalid number of input shapes");
    }

protected:
    std::shared_ptr<ov::Model> initOriginal() const override;
};

//  Quantized MatMul
//       FQ[I8]
//        Add
class MatMulBiasQuantizedFunction : public MatMulFunctionBase {
public:
    explicit MatMulBiasQuantizedFunction(const std::vector<PartialShape>& inputShapes,
                                         const std::vector<ov::element::Type>& precisions,
                                         MatMulType type)
        : MatMulFunctionBase(inputShapes, type, precisions) {
        OPENVINO_ASSERT(input_shapes.size() == 3, "Got invalid number of input shapes");
    }

protected:
    std::shared_ptr<ov::Model> initOriginal() const override;
};

//  Quantized MatMul  FQ[I8]
//       FQ[U8]    Reshape  <- To have only one sequence in Subgraph: MatMuL->FQ[U8]->MatMul->FQ[I8]
//            \     /
//             MatMul
//             FQ[I8]
class MatMulsQuantizedFunction : public MatMulFunctionBase {
public:
    explicit MatMulsQuantizedFunction(const std::vector<PartialShape>& inputShapes,
                                      const std::vector<ov::element::Type>& precisions,
                                      MatMulType type)
        : MatMulFunctionBase(inputShapes, type, precisions) {
        OPENVINO_ASSERT(input_shapes.size() == 3, "Got invalid number of input shapes");
    }

    std::set<size_t> get_constant_input_idces() const override {
        return matmul_type == MatMulType::FullyConnected ? std::set<size_t>{1, 2} : std::set<size_t>{};
    }

protected:
    std::shared_ptr<ov::Model> initOriginal() const override;
};

/// Minimal graph to test MatMul+Transpose combinations. Transpose location is specified via the position argument:
/// 0 - before the first MatMul input; 1 - before the second MatMul input; 2 - after the MatMul output.
/// Tokenized simply by starting subgraph,
//   in1        in2
//   Transpose  /
//         Matmul
//         Result
class Transpose0213MatMulFunction : public MatMulFunctionBase {
public:
    explicit Transpose0213MatMulFunction(const std::vector<PartialShape>& inputShapes, const std::vector<ov::element::Type>& precisions,
                                         MatMulType type, size_t position = 0)
    : MatMulFunctionBase(inputShapes, type, precisions), transpose_position(position)  {
        OPENVINO_ASSERT(input_shapes.size() == 2, "Got invalid number of input shapes");
        OPENVINO_ASSERT(input_shapes[0].size() == 4, "Only rank 4 input shapes are supported by this test");
        if (position == 1) {
            OPENVINO_ASSERT(input_shapes[1].size() == 4, "Only rank 4 input shapes are supported by this test");
            OPENVINO_ASSERT(type == MatMulType::MatMul, "Transpose on B input is not supported for FullyConnected type");
        }
        OPENVINO_ASSERT(transpose_position >=0 && transpose_position <= 2, "Got invalid transpose position");
    }
protected:
    std::shared_ptr<ov::Model> initOriginal() const override;
    size_t transpose_position;
};

class TransposeMatMulFunction : public MatMulFunctionBase {
public:
    explicit TransposeMatMulFunction(const std::vector<PartialShape>& inputShapes) : MatMulFunctionBase(inputShapes, MatMulType::MatMul) {
        OPENVINO_ASSERT(input_shapes.size() == 2, "Got invalid number of input shapes");
    }
protected:
    std::shared_ptr<ov::Model> initOriginal() const override;
};

class TransposeMatMulBiasFunction : public MatMulFunctionBase {
public:
    explicit TransposeMatMulBiasFunction(const std::vector<PartialShape>& inputShapes) : MatMulFunctionBase(inputShapes, MatMulType::MatMul) {
        OPENVINO_ASSERT(input_shapes.size() == 3, "Got invalid number of input shapes");
    }
protected:
    std::shared_ptr<ov::Model> initOriginal() const override;
};

class TransposeMulMatMulBiasFunction : public MatMulFunctionBase {
public:
    explicit TransposeMulMatMulBiasFunction(const std::vector<PartialShape>& inputShapes) : MatMulFunctionBase(inputShapes, MatMulType::MatMul) {
        OPENVINO_ASSERT(input_shapes.size() == 4, "Got invalid number of input shapes");
    }
protected:
    std::shared_ptr<ov::Model> initOriginal() const override;
};

//  Quantized MatMul  FQ[I8]
//       Softmax    Reshape  <- To have only one sequence in Subgraph: MatMuL->Softmax>FQ[U8]->MatMul->FQ[I8]
//        FQ[U8]     /
//             MatMul
//             FQ[I8]
class MatMulsQuantizedSoftmaxFunction : public MatMulFunctionBase {
public:
    explicit MatMulsQuantizedSoftmaxFunction(const std::vector<PartialShape>& inputShapes,
                                             const std::vector<ov::element::Type>& precisions,
                                             MatMulType type)
        : MatMulFunctionBase(inputShapes, type, precisions) {
        OPENVINO_ASSERT(input_shapes.size() == 3, "Got invalid number of input shapes");
    }

    std::set<size_t> get_constant_input_idces() const override {
        return matmul_type == MatMulType::FullyConnected ? std::set<size_t>{1, 2} : std::set<size_t>{};
    }

protected:
    std::shared_ptr<ov::Model> initOriginal() const override;
};

//         Param
//           |
//         MatMul
//           |
//        Softmax
class MatMulSoftmaxFunction : public MatMulFunctionBase {
public:
    explicit MatMulSoftmaxFunction(const std::vector<PartialShape>& inputShapes,
                                   const std::vector<ov::element::Type>& precisions,
                                   MatMulType type)
        : MatMulFunctionBase(inputShapes, type, precisions) {
        OPENVINO_ASSERT(input_shapes.size() == 2, "Got invalid number of input shapes");
    }

    std::set<size_t> get_constant_input_idces() const override {
        return matmul_type == MatMulType::FullyConnected ? std::set<size_t>{1} : std::set<size_t>{};
    }

protected:
    std::shared_ptr<ov::Model> initOriginal() const override;
    std::shared_ptr<ov::Model> initReference() const override;
};

//  MatMulBiasScalabilityFunction
//  Inputs: data0, W0, B0, W1, B1, ..., W{N-1}, B{N-1}
//  Constraint: M = 1 + 2*N  (data0 + (weight, bias) * N)
//  Tokenization per stage: MatMul -> Add(bias)
//  For FullyConnected: constant inputs are {2*i + 1 : W_i, 2*i + 2 : B_i}
//
//       data0
//         |
//       MatMul            <--- W0
//         |
//         Add             <--- B0
//         |
//       MatMul            <--- W1
//         |
//         Add             <--- B1
//         |
//         ...
//         |
//       MatMul            <--- W{N-1}
//         |
//         Add             <--- B{N-1}
//         |
//       Result
class MatMulBiasScalabilityFunction : public MatMulFunctionBase {
public:
    explicit MatMulBiasScalabilityFunction(const std::vector<PartialShape>& inputShapes,
                                           const std::vector<ov::element::Type>& precisions,
                                           MatMulType type,
                                           size_t num_repetitions = 2)
        : MatMulFunctionBase(inputShapes, type, precisions) {
        const size_t M = input_shapes.size();
        OPENVINO_ASSERT(M >= 3 && ((M - 1) % 2 == 0),
            std::string("Got invalid number of input shapes: expected 1 + 2*N (data0 + (weight, bias) * N), got ")
            + std::to_string(M));
        // Derive the true stage count from shapes to avoid desync with test params:
        m_num_repetitions = (M - 1) / 2;
    }

    std::set<size_t> get_constant_input_idces() const override {
        std::set<size_t> constant_idces;
        if (matmul_type == MatMulType::FullyConnected) {
            // Chained layout: W_i at 2*i+1, B_i at 2*i+2
            for (size_t i = 0; i < m_num_repetitions; ++i) {
                constant_idces.insert(2 * i + 1);  // weight indices
                constant_idces.insert(2 * i + 2);  // bias indices
            }
        }
        return constant_idces;
    }

protected:
    std::shared_ptr<ov::Model> initOriginal() const override;
    std::shared_ptr<ov::Model> initReference() const override;

private:
    size_t m_num_repetitions{};
};

//         MatMul
//           |   |
//           |  Eltwise chain
//            \     /
//              Add
class MatMulEltwiseChainFunction : public MatMulFunctionBase {
public:
    explicit MatMulEltwiseChainFunction(const std::vector<PartialShape>& inputShapes,
                                const std::vector<ov::element::Type>& precisions,
                                MatMulType type)
        : MatMulFunctionBase(inputShapes, type, precisions) {
        OPENVINO_ASSERT(input_shapes.size() == 2, "Got invalid number of input shapes");
    }

protected:
    std::shared_ptr<ov::Model> initOriginal() const override;
};

//         MatMul
//           |   |
//           |  Eltwise chain
//            \     /
//              Add
//               |
//             MatMul
//               |
//        Eltwise chain
class MatMulEltwiseChainCascadeFunction : public MatMulFunctionBase {
public:
    explicit MatMulEltwiseChainCascadeFunction(const std::vector<PartialShape>& inputShapes,
                                               const std::vector<ov::element::Type>& precisions,
                                               MatMulType type)
        : MatMulFunctionBase(inputShapes, type, precisions) {
        OPENVINO_ASSERT(input_shapes.size() == 3, "Got invalid number of input shapes");
    }

    std::set<size_t> get_constant_input_idces() const override {
        return matmul_type == MatMulType::FullyConnected ? std::set<size_t>{1, 2} : std::set<size_t>{};
    }

protected:
    std::shared_ptr<ov::Model> initOriginal() const override;
};

}  // namespace snippets
}  // namespace test
}  // namespace ov
