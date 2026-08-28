!boot_context = !cheddar.boot_context
!ciphertext = !cheddar.ciphertext
!evk_map = !cheddar.evk_map
!user_interface = !cheddar.user_interface

module attributes {
  cheddar.boot.num_cts = 4 : i64,
  cheddar.boot.num_stc = 3 : i64,
  ckks.schemeParam = #ckks.scheme_param<logN = 16, Q = [1125899908022273, 1099515691009, 1099523555329, 1099525128193, 1099526176769, 1099529060353, 1099535220737, 1099536138241, 1099537580033, 1099538104321, 1099540725761, 1099540856833, 1099543085057, 36028797019488257, 36028797023420417, 36028797024206849, 36028797025124353, 36028797032202241, 36028797033644033, 36028797037576193, 36028797048324097, 36028797048586241, 36028797049896961, 36028797051863041, 36028797053698049, 36028797054222337], P = [72057594038321153, 72057594040680449, 72057594042646529, 72057594047889409, 72057594057195521, 72057594058375169, 72057594058899457], logDefaultScale = 40>,
  scheme.actual_slot_count = 1024 : i64,
  scheme.requested_slot_count = 1024 : i64
} {
  // The rotation is here so that key generation emits a per-distance
  // `prepare_rot_key` alongside the bootstrap's bulk request: the two take
  // different PrepareRotationKey overloads, and on Cyclops both name the
  // secret the keys are built for.
  func.func @kernel(%ctx: !boot_context, %ui: !user_interface,
                    %ct: tensor<!ciphertext>, %evk: !evk_map)
      -> tensor<!ciphertext> {
    %rot_dest = tensor.empty() : tensor<!ciphertext>
    %rot = cheddar.hrot %ctx, %ui, %ct, %rot_dest
        {level = 1 : i64, static_distance = 7 : i64}
        : (!boot_context, !user_interface, tensor<!ciphertext>, tensor<!ciphertext>) -> tensor<!ciphertext>
    %dest = tensor.empty() : tensor<!ciphertext>
    %result = cheddar.boot %ctx, %rot, %evk, %dest : (!boot_context, tensor<!ciphertext>, !evk_map, tensor<!ciphertext>) -> tensor<!ciphertext>
    return %result : tensor<!ciphertext>
  }
}
