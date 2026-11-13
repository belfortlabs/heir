// RUN: heir-opt --add-client-interface="min-slot-count=2048" %s | FileCheck %s

// The result of a stride-2 conv on 1x8x16x16 data is pixel-shuffled: its layout
// is the row-major 1x8x8x8 layout, then the shuffle. The decrypt function
// unpacks through the composed layout.

!ct_ty = !secret.secret<tensor<1x2048xf32>>
#input_layout = #tensor_ext.layout<"{ [i0, i1, i2, i3] -> [ct, slot] : i0 = 0 and ct = 0 and (-256i1 - 16i2 - i3 + slot) mod 2048 = 0 and 0 <= i1 <= 7 and 0 <= i2 <= 15 and 0 <= i3 <= 15 and 0 <= slot <= 2047 }">
#row_major = #tensor_ext.layout<"{ [i0, i1, i2, i3] -> [ct, slot] : i0 = 0 and ct = 0 and (-64i1 - 8i2 - i3 + slot) mod 512 = 0 and 0 <= i1 <= 7 and 0 <= i2 <= 7 and 0 <= i3 <= 7 and 0 <= slot <= 2047 }">
#shuffle = #tensor_ext.layout<"{ [i0, i1] -> [ct, slot] : exists (e1, e2, e3, e4, e5: i0 = 0 and ct = 0 and 0 <= i1 <= 2047 and 0 <= slot <= 2047 and 512*floor((slot)/512) >= -511 + i1 and 0 <= e1 <= 7 and 2e2 >= slot and -7 + i1 + 64slot - 512*floor((slot)/512) - 8e1 <= 128e2 <= i1 + 64slot - 512*floor((slot)/512) - 8e1 and -3 - slot + 2e2 <= 4e3 <= -slot + 2e2 and -1 - slot + 4e1 + 2e2 - 4e3 <= 2e4 <= -slot + 4e1 + 2e2 - 4e3 and -3 - slot + 2e2 <= 4e5 <= -slot + 2e2 and -1 - 2i1 - 127slot + 512*floor((slot)/512) + 16e1 + 256e2 - 16e4 <= 256e5 <= -2i1 - 127slot + 512*floor((slot)/512) + 16e1 + 256e2 - 16e4) }">
#input_type = #tensor_ext.original_type<originalType = tensor<1x8x16x16xf32>, layout = #input_layout>
#result_type = #tensor_ext.original_type<originalType = tensor<1x8x8x8xf32>, layout = [#row_major, #shuffle]>

// CHECK: func.func @strided_conv__decrypt__result0
// CHECK: secret.reveal
// CHECK-NOT: tensor.insert {{.*}} : tensor<1x2048xf32>
// CHECK: scf.for
// CHECK: tensor.extract {{.*}} : tensor<1x2048xf32>
// CHECK: tensor.insert {{.*}} : tensor<1x8x8x8xf32>
// CHECK: return {{.*}} : tensor<1x8x8x8xf32>
func.func @strided_conv(
    %arg0: !ct_ty {tensor_ext.original_type = #input_type}
) -> (!ct_ty {tensor_ext.original_type = #result_type}) {
  %0 = secret.generic(%arg0: !ct_ty) {
  ^body(%pt_arg0: tensor<1x2048xf32>):
    secret.yield %pt_arg0 : tensor<1x2048xf32>
  } -> !ct_ty
  return %0 : !ct_ty
}
