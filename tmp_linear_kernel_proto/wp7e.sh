# Prints heir-opt output for the lit tests whose expectations changed in WP7.
cd ~/sky_workdir
bash tmp_linear_kernel_proto/ir.sh tests/Transforms/convert_to_ciphertext_semantics/strided_matvec.mlir --strided-layout-propagation=min-slot-count=16 --convert-to-ciphertext-semantics=min-slot-count=16
cut -c1-250 ~/ir.out
echo ===== fusion
bash tmp_linear_kernel_proto/ir.sh tests/Transforms/linalg_canonicalizations/conv_pool_fusion.mlir --linalg-canonicalizations --split-input-file
grep -n 'func.func\|linalg\.\|arith.constant\|return\|// -----\|arith\.' ~/ir.out | cut -c1-260
echo ===== fusion outs
bash tmp_linear_kernel_proto/ir.sh tests/Transforms/linalg_canonicalizations/conv_pool_fusion_outs.mlir --linalg-canonicalizations --canonicalize --split-input-file
grep -n 'func.func\|linalg\.\|arith.constant\|return\|// -----\|arith\.' ~/ir.out | cut -c1-260
