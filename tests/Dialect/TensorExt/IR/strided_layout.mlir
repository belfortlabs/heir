// RUN: heir-opt %s | FileCheck %s

// CHECK: #[[ROW_MAJOR:.*]] = #tensor_ext.strided_layout<shape = [3, 4, 5], slots = 64, digits = [(0, 1, 0, 20), (1, 1, 0, 5), (2, 1, 0, 1)]>
#row_major = #tensor_ext.strided_layout<shape = [3, 4, 5], slots = 64, digits = [(0, 1, 0, 20), (1, 1, 0, 5), (2, 1, 0, 1)]>

// CHECK: #[[GAPPED:.*]] = #tensor_ext.strided_layout<shape = [8, 4, 4], slots = 64, digits = [(1, 1, 0, 16), (2, 1, 0, 2), (0, 1, 2, 1), (0, 2, 2, 8), (0, 4, 0, 64)], offset = 3>
#gapped = #tensor_ext.strided_layout<shape = [8, 4, 4], slots = 64,
    digits = [(1, 1, 0, 16), (2, 1, 0, 2), (0, 1, 2, 1), (0, 2, 2, 8), (0, 4, 0, 64)],
    offset = 3>

// CHECK: #[[PERIODIC:.*]] = #tensor_ext.strided_layout<shape = [5], slots = 32, digits = [(0, 1, 0, 1)], period = 8>
#periodic = #tensor_ext.strided_layout<shape = [5], slots = 32, digits = [(0, 1, 0, 1)], period = 8>

// CHECK: #[[SCALAR:.*]] = #tensor_ext.strided_layout<shape = [1], slots = 32, digits = []>
#scalar = #tensor_ext.strided_layout<shape = [1], slots = 32, digits = []>

// CHECK: @row_major(tensor<3x4x5xi32> {foo.bar = #[[ROW_MAJOR]]})
func.func private @row_major(tensor<3x4x5xi32> {foo.bar = #row_major})
// CHECK: @gapped(tensor<8x4x4xi32> {foo.bar = #[[GAPPED]]})
func.func private @gapped(tensor<8x4x4xi32> {foo.bar = #gapped})
// CHECK: @periodic(tensor<5xi32> {foo.bar = #[[PERIODIC]]})
func.func private @periodic(tensor<5xi32> {foo.bar = #periodic})
// CHECK: @scalar(tensor<1xi32> {foo.bar = #[[SCALAR]]})
func.func private @scalar(tensor<1xi32> {foo.bar = #scalar})
