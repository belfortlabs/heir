// RUN: heir-opt --cheddar-elide-out-param-copies %s | FileCheck %s

// CHECK: func.func @forward(%[[CTX:.*]]: !context, %[[IN:.*]]: memref<!ciphertext>, %[[OUT:.*]]: memref<!ciphertext> {bufferize.result})
// CHECK-NEXT: cheddar.neg %[[CTX]], %[[IN]], %[[OUT]]
// CHECK-NEXT: return
func.func @forward(%ctx: !cheddar.context, %input: memref<!cheddar.ciphertext>, %out: memref<!cheddar.ciphertext> {bufferize.result}) {
  %tmp = memref.alloc() : memref<!cheddar.ciphertext>
  cheddar.neg %ctx, %input, %tmp : (!cheddar.context, memref<!cheddar.ciphertext>, memref<!cheddar.ciphertext>) -> ()
  memref.copy %tmp, %out : memref<!cheddar.ciphertext> to memref<!cheddar.ciphertext>
  return
}

// The temporary is still modified after the copy: keep the copy.
// CHECK: func.func @used_after_copy
// CHECK: memref.alloc
// CHECK: memref.copy
func.func @used_after_copy(%ctx: !cheddar.context, %input: memref<!cheddar.ciphertext>, %out: memref<!cheddar.ciphertext> {bufferize.result}) {
  %tmp = memref.alloc() : memref<!cheddar.ciphertext>
  cheddar.neg %ctx, %input, %tmp : (!cheddar.context, memref<!cheddar.ciphertext>, memref<!cheddar.ciphertext>) -> ()
  memref.copy %tmp, %out : memref<!cheddar.ciphertext> to memref<!cheddar.ciphertext>
  cheddar.neg %ctx, %tmp, %tmp : (!cheddar.context, memref<!cheddar.ciphertext>, memref<!cheddar.ciphertext>) -> ()
  return
}

// Only out-params are known to be fresh, exclusively owned buffers.
// CHECK: func.func @plain_argument
// CHECK: memref.alloc
// CHECK: memref.copy
func.func @plain_argument(%ctx: !cheddar.context, %input: memref<!cheddar.ciphertext>, %out: memref<!cheddar.ciphertext>) {
  %tmp = memref.alloc() : memref<!cheddar.ciphertext>
  cheddar.neg %ctx, %input, %tmp : (!cheddar.context, memref<!cheddar.ciphertext>, memref<!cheddar.ciphertext>) -> ()
  memref.copy %tmp, %out : memref<!cheddar.ciphertext> to memref<!cheddar.ciphertext>
  return
}
