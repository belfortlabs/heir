// RUN: heir-opt '--cheddar-configure-crypto-context=entry-function=main use-cyclops-runtime=true' --split-input-file --verify-diagnostics %s

!boot_context = !cheddar.boot_context
!ciphertext = !cheddar.ciphertext
!evk_map = !cheddar.evk_map
!ui = !cheddar.user_interface

// A bootstrapping program needs the bootstrap chain of its parameter set.
module attributes {
  cheddar.parameter_set = #cheddar.parameter_set<logN = 16, logScale = 35, mainPrimes = [969146369, 1172439041], auxPrimes = [964558849], terminalPrimes = [30539777, 32899073], levelConfig = [0, 2, 2, 1], wordBits = 32>,
  scheme.requested_slot_count = 8 : i64
} {
  // expected-error@+1 {{program bootstraps, but the parameter set carries no bootstrap chain}}
  func.func @main(%ctx: !boot_context, %ct: tensor<!ciphertext>, %evk: !evk_map) -> tensor<!ciphertext> {
    %dest = bufferization.alloc_tensor() : tensor<!ciphertext>
    %result = cheddar.boot %ctx, %ct, %evk, %dest : (!boot_context, tensor<!ciphertext>, !evk_map, tensor<!ciphertext>) -> tensor<!ciphertext>
    return %result : tensor<!ciphertext>
  }
}

// -----

!boot_context = !cheddar.boot_context
!ciphertext = !cheddar.ciphertext
!evk_map = !cheddar.evk_map
!ui = !cheddar.user_interface

// Cyclops accepts a default encryption level anywhere on SlotToCoeff's
// levels: here the circuit starts SlotToCoeff at level 2 and lands at level
// 1, so level 0 is too low.
module attributes {
  cheddar.bootstrap_config = #cheddar.bootstrap_config<numCtsLevels = 1, numStcLevels = 1, numEvalModLevels = 0>,
  cheddar.parameter_set = #cheddar.parameter_set<logN = 16, logScale = 35, mainPrimes = [969146369, 1172439041, 958136321, 1070727169], auxPrimes = [964558849], terminalPrimes = [30539777, 32899073], levelConfig = [0, 2, 2, 1, 3, 1, 4, 1], wordBits = 32, defaultEncryptionLevel = 0>,
  scheme.requested_slot_count = 8 : i64
} {
  // expected-error@+1 {{the default encryption level 0 does not fit the bootstrap, whose SlotToCoeff runs from level 2 down to level 1}}
  func.func @main(%ctx: !boot_context, %ct: tensor<!ciphertext>, %evk: !evk_map) -> tensor<!ciphertext> {
    %dest = bufferization.alloc_tensor() : tensor<!ciphertext>
    %result = cheddar.boot %ctx, %ct, %evk, %dest : (!boot_context, tensor<!ciphertext>, !evk_map, tensor<!ciphertext>) -> tensor<!ciphertext>
    return %result : tensor<!ciphertext>
  }
}
