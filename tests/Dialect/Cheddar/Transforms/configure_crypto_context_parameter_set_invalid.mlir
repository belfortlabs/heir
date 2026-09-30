// RUN: heir-opt --cheddar-configure-crypto-context=entry-function=main --split-input-file --verify-diagnostics %s

!boot_context = !cheddar.boot_context
!ciphertext = !cheddar.ciphertext
!context = !cheddar.context
!evk_map = !cheddar.evk_map
!ui = !cheddar.user_interface

// scale-snu's EvalMod always takes eight levels.
module attributes {
  cheddar.bootstrap_config = #cheddar.bootstrap_config<numCtsLevels = 1, numStcLevels = 1, numEvalModLevels = 0>,
  cheddar.parameter_set = #cheddar.parameter_set<logN = 16, logScale = 35, mainPrimes = [969146369, 1172439041, 958136321, 1070727169], auxPrimes = [964558849], terminalPrimes = [30539777, 32899073], levelConfig = [0, 2, 2, 1, 3, 1, 4, 1], wordBits = 32, defaultEncryptionLevel = 2>,
  scheme.requested_slot_count = 8 : i64
} {
  // expected-error@+1 {{scale-snu CHEDDAR's EvalMod consumes 8 levels, but the parameter set reserves 0}}
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

// scale-snu's BootContext needs the default encryption level at the start of
// SlotToCoeff (level 10 = 19 - 1 CtS - 8 EvalMod), not at its end (level 9).
module attributes {
  cheddar.bootstrap_config = #cheddar.bootstrap_config<numCtsLevels = 1, numStcLevels = 1, numEvalModLevels = 8>,
  cheddar.parameter_set = #cheddar.parameter_set<logN = 16, logScale = 40, mainPrimes = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20], auxPrimes = [21], defaultEncryptionLevel = 9>,
  scheme.requested_slot_count = 8 : i64
} {
  // expected-error@+1 {{the default encryption level 9 does not fit the bootstrap, whose SlotToCoeff runs from level 10 down to level 9}}
  func.func @main(%ctx: !boot_context, %ct: tensor<!ciphertext>, %evk: !evk_map) -> tensor<!ciphertext> {
    %dest = bufferization.alloc_tensor() : tensor<!ciphertext>
    %result = cheddar.boot %ctx, %ct, %evk, %dest : (!boot_context, tensor<!ciphertext>, !evk_map, tensor<!ciphertext>) -> tensor<!ciphertext>
    return %result : tensor<!ciphertext>
  }
}
