// RUN: heir-opt --verify-diagnostics --split-input-file %s

// expected-error@below {{expected a digit (axis, div, mod, stride)}}
#layout = #tensor_ext.strided_layout<shape = [4], slots = 8, digits = [(0, 1, 1)]>
func.func private @test_fn(tensor<4xi32> {foo.bar = #layout})

// -----

// expected-error@below {{digit axis 1 is out of bounds for 1 axes}}
#layout = #tensor_ext.strided_layout<shape = [4], slots = 8, digits = [(1, 1, 0, 1)]>
func.func private @test_fn(tensor<4xi32> {foo.bar = #layout})

// -----

// expected-error@below {{period must divide slots}}
#layout = #tensor_ext.strided_layout<shape = [4], slots = 8, digits = [(0, 1, 0, 1)], period = 3>
func.func private @test_fn(tensor<4xi32> {foo.bar = #layout})

// -----

// expected-error@below {{a periodic layout must fit within one period}}
#layout = #tensor_ext.strided_layout<shape = [6], slots = 8, digits = [(0, 1, 0, 1)], period = 4>
func.func private @test_fn(tensor<6xi32> {foo.bar = #layout})

// -----

// expected-error@below {{expected 'offset' or 'period'}}
#layout = #tensor_ext.strided_layout<shape = [4], slots = 8, digits = [(0, 1, 0, 1)], gap = 2>
func.func private @test_fn(tensor<4xi32> {foo.bar = #layout})
