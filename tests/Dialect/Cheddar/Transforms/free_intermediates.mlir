// RUN: heir-opt --cheddar-free-intermediates %s | FileCheck %s

!ciphertext = !cheddar.ciphertext
!context = !cheddar.context
!eval_key = !cheddar.eval_key

// The local payload intermediate (%alloc, the mult result) is freed right after
// its last use (the relinearize), so its GPU buffer is released at last use
// rather than at function-scope cleanup. The returned out-param (%arg2) is a
// function argument, not a memref.alloc, so it is never freed.

// CHECK: @free_intermediate
// CHECK: %[[A:.*]] = memref.alloc
// CHECK: cheddar.mult %{{.*}}, %[[A]]
// CHECK: cheddar.relinearize %{{.*}}, %[[A]]
// CHECK-NEXT: memref.dealloc %[[A]]
// CHECK-NOT: memref.dealloc %arg2
func.func @free_intermediate(%ctx: !context, %arg0: memref<!ciphertext>,
    %arg1: memref<!ciphertext>, %evk: !eval_key,
    %arg2: memref<!ciphertext> {bufferize.result}) {
  %alloc = memref.alloc() {alignment = 64 : i64} : memref<!ciphertext>
  cheddar.mult %ctx, %arg0, %arg1, %alloc : (!context, memref<!ciphertext>, memref<!ciphertext>, memref<!ciphertext>) -> ()
  cheddar.relinearize %ctx, %alloc, %evk, %arg2 : (!context, memref<!ciphertext>, !eval_key, memref<!ciphertext>) -> ()
  return
}

// A plain float buffer is a scope-bound stack array with nothing to free: the
// pass leaves it alone (only CHEDDAR payload buffers are freed).
// CHECK: @no_free_for_float
// CHECK-NOT: memref.dealloc
func.func @no_free_for_float(%arg0: memref<8xf32>) {
  %alloc = memref.alloc() : memref<8xf32>
  %c0 = arith.constant 0 : index
  %v = memref.load %alloc[%c0] : memref<8xf32>
  memref.store %v, %arg0[%c0] : memref<8xf32>
  return
}
