// RUN: heir-opt %s --annotate-module="backend=cheddar scheme=ckks" --mlir-to-ckks="min-slot-count=4096 enable-split-preprocessing=true" --scheme-to-cheddar="entry-function=matvec runtime=cyclops" --cheddar-to-emitc --cheddar-emitc-entry-interface=runtime=cyclops > %t
// RUN: heir-translate %t --mlir-to-cpp --file-id=client_source | FileCheck %s --check-prefix=CLIENT --implicit-check-not=LinearTransform --implicit-check-not=__constant_8x4xf32
// RUN: heir-translate %t --mlir-to-cpp --file-id=server_source | FileCheck %s --check-prefix=SERVER --implicit-check-not=UserInterface

// CLIENT: ClientContext<word>::Create
// CLIENT: EncryptedInputs Encrypt
// SERVER: LinearTransform<word>
// SERVER: Preprocess
// SERVER: heir::cyclops::keyRequest

module {
  func.func @matvec(%arg0 : tensor<4xf32> {secret.secret}) -> tensor<8xf32> {
    %matrix = arith.constant dense<[[0.0, 1.0, 2.0, 3.0],
                                    [4.0, 5.0, 6.0, 7.0],
                                    [8.0, 9.0, 10.0, 11.0],
                                    [12.0, 13.0, 14.0, 15.0],
                                    [16.0, 17.0, 18.0, 19.0],
                                    [20.0, 21.0, 22.0, 23.0],
                                    [24.0, 25.0, 26.0, 27.0],
                                    [28.0, 29.0, 30.0, 31.0]]> : tensor<8x4xf32>
    %out = arith.constant dense<0.0> : tensor<8xf32>
    %0 = linalg.matvec ins(%matrix, %arg0 : tensor<8x4xf32>, tensor<4xf32>) outs(%out : tensor<8xf32>) -> tensor<8xf32>
    return %0 : tensor<8xf32>
  }
}
