// RUN: heir-opt --cheddar-fuse-ops %s | FileCheck %s

!ct = !cheddar.ciphertext

// CHECK: @fuse_hmult_rescale
func.func @fuse_hmult_rescale(
    %ctx: !cheddar.context, %lhs: tensor<!ct>, %rhs: tensor<!ct>,
    %key: !cheddar.eval_key) -> tensor<!ct> {
  // CHECK-NOT: cheddar.mult
  // CHECK-NOT: cheddar.relinearize
  // CHECK-NOT: cheddar.rescale
  // CHECK: cheddar.hmult
  // CHECK-NOT: rescale = false
  %d0 = bufferization.alloc_tensor() : tensor<!ct>
  %mult = cheddar.mult %ctx, %lhs, %rhs, %d0 : (!cheddar.context, tensor<!ct>, tensor<!ct>, tensor<!ct>) -> tensor<!ct>
  %d1 = bufferization.alloc_tensor() : tensor<!ct>
  %relin = cheddar.relinearize %ctx, %mult, %key, %d1 : (!cheddar.context, tensor<!ct>, !cheddar.eval_key, tensor<!ct>) -> tensor<!ct>
  %d2 = bufferization.alloc_tensor() : tensor<!ct>
  %result = cheddar.rescale %ctx, %relin, %d2 : (!cheddar.context, tensor<!ct>, tensor<!ct>) -> tensor<!ct>
  return %result : tensor<!ct>
}

// CHECK: @fuse_hmult_no_rescale
func.func @fuse_hmult_no_rescale(
    %ctx: !cheddar.context, %lhs: tensor<!ct>, %rhs: tensor<!ct>,
    %key: !cheddar.eval_key) -> tensor<!ct> {
  // CHECK-NOT: cheddar.mult
  // CHECK-NOT: cheddar.relinearize
  // CHECK: cheddar.hmult
  // CHECK-SAME: rescale = false
  %d0 = bufferization.alloc_tensor() : tensor<!ct>
  %mult = cheddar.mult %ctx, %lhs, %rhs, %d0 : (!cheddar.context, tensor<!ct>, tensor<!ct>, tensor<!ct>) -> tensor<!ct>
  %d1 = bufferization.alloc_tensor() : tensor<!ct>
  %result = cheddar.relinearize %ctx, %mult, %key, %d1 : (!cheddar.context, tensor<!ct>, !cheddar.eval_key, tensor<!ct>) -> tensor<!ct>
  return %result : tensor<!ct>
}

// CHECK: @fuse_hmult_relinearize_rescale
func.func @fuse_hmult_relinearize_rescale(
    %ctx: !cheddar.context, %lhs: tensor<!ct>, %rhs: tensor<!ct>,
    %key: !cheddar.eval_key) -> tensor<!ct> {
  // CHECK-NOT: cheddar.mult
  // CHECK-NOT: cheddar.relinearize_rescale
  // CHECK: cheddar.hmult
  %d0 = bufferization.alloc_tensor() : tensor<!ct>
  %mult = cheddar.mult %ctx, %lhs, %rhs, %d0 : (!cheddar.context, tensor<!ct>, tensor<!ct>, tensor<!ct>) -> tensor<!ct>
  %d1 = bufferization.alloc_tensor() : tensor<!ct>
  %result = cheddar.relinearize_rescale %ctx, %mult, %key, %d1 : (!cheddar.context, tensor<!ct>, !cheddar.eval_key, tensor<!ct>) -> tensor<!ct>
  return %result : tensor<!ct>
}

// CHECK: @fuse_rotation_and_conjugation
func.func @fuse_rotation_and_conjugation(
    %ctx: !cheddar.context, %evk: !cheddar.evk_map,
    %input: tensor<!ct>, %other: tensor<!ct>) -> (tensor<!ct>, tensor<!ct>) {
  // CHECK: cheddar.hrot_add
  // CHECK-SAME: distance = 3
  // CHECK-SAME: level = 3
  // CHECK: cheddar.hconj_add
  %r0 = bufferization.alloc_tensor() : tensor<!ct>
  %rotated = cheddar.hrot %ctx, %evk, %input, %r0 {level = 3 : i64, static_distance = 3 : i64} : (!cheddar.context, !cheddar.evk_map, tensor<!ct>, tensor<!ct>) -> tensor<!ct>
  %r1 = bufferization.alloc_tensor() : tensor<!ct>
  %rotated_sum = cheddar.add %ctx, %rotated, %other, %r1 : (!cheddar.context, tensor<!ct>, tensor<!ct>, tensor<!ct>) -> tensor<!ct>
  %c0 = bufferization.alloc_tensor() : tensor<!ct>
  %conjugated = cheddar.hconj %ctx, %evk, %input, %c0 : (!cheddar.context, !cheddar.evk_map, tensor<!ct>, tensor<!ct>) -> tensor<!ct>
  %c1 = bufferization.alloc_tensor() : tensor<!ct>
  %conjugated_sum = cheddar.add %ctx, %conjugated, %other, %c1 : (!cheddar.context, tensor<!ct>, tensor<!ct>, tensor<!ct>) -> tensor<!ct>
  return %rotated_sum, %conjugated_sum : tensor<!ct>, tensor<!ct>
}

// Relinearization commutes with add_plain when all operations share a context.
// CHECK: @hoist_relinearize
func.func @hoist_relinearize(
    %ctx: !cheddar.context, %lhs: tensor<!ct>, %rhs: tensor<!ct>,
    %plain: tensor<!cheddar.plaintext>, %key: !cheddar.eval_key) -> tensor<!ct> {
  // CHECK-NOT: cheddar.mult
  // CHECK-NOT: cheddar.relinearize
  // CHECK: cheddar.hmult
  // CHECK: cheddar.add_plain
  %d0 = bufferization.alloc_tensor() : tensor<!ct>
  %mult = cheddar.mult %ctx, %lhs, %rhs, %d0 : (!cheddar.context, tensor<!ct>, tensor<!ct>, tensor<!ct>) -> tensor<!ct>
  %d1 = bufferization.alloc_tensor() : tensor<!ct>
  %added = cheddar.add_plain %ctx, %mult, %plain, %d1 : (!cheddar.context, tensor<!ct>, tensor<!cheddar.plaintext>, tensor<!ct>) -> tensor<!ct>
  %d2 = bufferization.alloc_tensor() : tensor<!ct>
  %result = cheddar.relinearize %ctx, %added, %key, %d2 : (!cheddar.context, tensor<!ct>, !cheddar.eval_key, tensor<!ct>) -> tensor<!ct>
  return %result : tensor<!ct>
}

// Fusing across different contexts would silently change semantics.
// CHECK: @do_not_fuse_different_contexts
func.func @do_not_fuse_different_contexts(
    %ctx0: !cheddar.context, %ctx1: !cheddar.context,
    %evk: !cheddar.evk_map, %input: tensor<!ct>,
    %other: tensor<!ct>) -> tensor<!ct> {
  // CHECK: cheddar.hrot
  // CHECK: cheddar.add
  // CHECK-NOT: cheddar.hrot_add
  %d0 = bufferization.alloc_tensor() : tensor<!ct>
  %rotated = cheddar.hrot %ctx0, %evk, %input, %d0 {level = 2 : i64, static_distance = 2 : i64} : (!cheddar.context, !cheddar.evk_map, tensor<!ct>, tensor<!ct>) -> tensor<!ct>
  %d1 = bufferization.alloc_tensor() : tensor<!ct>
  %result = cheddar.add %ctx1, %rotated, %other, %d1 : (!cheddar.context, tensor<!ct>, tensor<!ct>, tensor<!ct>) -> tensor<!ct>
  return %result : tensor<!ct>
}

// A linear transform prepared by the function that applies it, here twice, is
// evaluated directly at each use, so its encoded diagonals are not kept on the
// device between the uses.

// CHECK: @apply_prepared_directly
// CHECK-NOT: cheddar.prepare_linear_transform
// CHECK-COUNT-2: cheddar.linear_transform %{{.*}} {bs = 2 : i64, diagonal_indices = array<i32: 0, 3>, gs = 1 : i64, level = 5 : i64}
// CHECK-NOT: cheddar.apply_prepared_linear_transform
func.func @apply_prepared_directly(
    %ctx: !cheddar.boot_context, %evk: !cheddar.evk_map, %a: tensor<1x!ct>,
    %b: tensor<1x!ct>, %diagonals: tensor<2x8xf32>) -> (tensor<1x!ct>, tensor<1x!ct>) {
  %d = tensor.empty() : tensor<!cheddar.linear_transform>
  %t = cheddar.prepare_linear_transform %ctx, %diagonals, %d {bs = 2 : i64, diagonal_indices = array<i32: 0, 3>, gs = 1 : i64, level = 5 : i64, width = 8 : i64} : (!cheddar.boot_context, tensor<2x8xf32>, tensor<!cheddar.linear_transform>) -> tensor<!cheddar.linear_transform>
  %o0 = tensor.empty() : tensor<1x!ct>
  %r0 = cheddar.apply_prepared_linear_transform %ctx, %a, %evk, %t, %o0 : (!cheddar.boot_context, tensor<1x!ct>, !cheddar.evk_map, tensor<!cheddar.linear_transform>, tensor<1x!ct>) -> tensor<1x!ct>
  %o1 = tensor.empty() : tensor<1x!ct>
  %r1 = cheddar.apply_prepared_linear_transform %ctx, %b, %evk, %t, %o1 : (!cheddar.boot_context, tensor<1x!ct>, !cheddar.evk_map, tensor<!cheddar.linear_transform>, tensor<1x!ct>) -> tensor<1x!ct>
  return %r0, %r1 : tensor<1x!ct>, tensor<1x!ct>
}
