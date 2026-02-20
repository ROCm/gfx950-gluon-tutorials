	.amdgcn_target "amdgcn-amd-amdhsa--gfx950"
	.amdhsa_code_object_version 5
	.text
	.globl	v8_beyond_hotloop               ; -- Begin function v8_beyond_hotloop
	.p2align	8
	.type	v8_beyond_hotloop,@function
v8_beyond_hotloop:                      ; @v8_beyond_hotloop
.Lfunc_begin0:
	.cfi_sections .debug_frame
	.cfi_startproc
; %bb.6:
s_load_dwordx2 s[2:3], s[0:1], 0x0
s_load_dwordx8 s[4:11], s[0:1], 0x8
s_load_dwordx4 s[12:15], s[0:1], 0x28
s_waitcnt lgkmcnt(0)
s_branch .LBB0_0
.p2align 8
; %bb.7:
.LBB0_0:
v_mov_b32_e32 v34, v0
s_nop 0
v_readfirstlane_b32 s53, v34
s_bfe_u32 s1, s53, 0x20006
s_add_i32 s0, s8, 0xff
s_ashr_i32 s8, s0, 31
s_lshr_b32 s8, s8, 24
s_add_i32 s0, s0, s8
s_ashr_i32 s0, s0, 8
s_add_i32 s8, s9, 0xff
s_ashr_i32 s9, s8, 31
s_lshr_b32 s9, s9, 24
s_add_i32 s8, s8, s9
s_ashr_i32 s8, s8, 8
s_ashr_i32 s9, s16, 31
s_lshr_b32 s9, s9, 29
s_add_i32 s9, s16, s9
s_ashr_i32 s9, s9, 3
s_lshl_b32 s14, s16, 5
s_mulk_i32 s9, 0xff01
s_add_i32 s9, s9, s14
s_lshl_b32 s14, s8, 2
s_xor_b32 s8, s9, s8
s_ashr_i32 s8, s8, 31
s_abs_i32 s15, s9
s_abs_i32 s16, s14
v_cvt_f32_u32_e32 v0, s16
v_rcp_iflag_f32_e32 v0, v0
s_nop 0
v_mul_f32_e32 v0, 0x4f7ffffe, v0
v_cvt_u32_f32_e32 v0, v0
s_mov_b32 s24, 0
s_sub_i32 s17, 0, s16
v_readfirstlane_b32 s18, v0
s_mul_i32 s17, s17, s18
s_mul_hi_u32 s17, s18, s17
s_add_i32 s18, s18, s17
s_mul_hi_u32 s17, s15, s18
s_mul_i32 s18, s17, s16
s_sub_i32 s15, s15, s18
s_add_i32 s18, s17, 1
s_sub_i32 s19, s15, s16
s_cmp_ge_u32 s15, s16
s_cselect_b32 s17, s18, s17
s_cselect_b32 s15, s19, s15
s_add_i32 s18, s17, 1
s_cmp_ge_u32 s15, s16
s_cselect_b32 s15, s18, s17
s_xor_b32 s15, s15, s8
s_sub_i32 s8, s15, s8
s_lshl_b32 s15, s8, 2
s_sub_i32 s0, s0, s15
s_min_i32 s0, s0, 4
s_mul_i32 s8, s8, s14
s_sub_i32 s8, s9, s8
s_xor_b32 s9, s8, s0
s_ashr_i32 s9, s9, 31
s_abs_i32 s14, s8
s_abs_i32 s16, s0
v_cvt_f32_u32_e32 v0, s16
v_rcp_iflag_f32_e32 v0, v0
s_nop 0
v_mul_f32_e32 v0, 0x4f7ffffe, v0
v_cvt_u32_f32_e32 v0, v0
s_sub_i32 s17, 0, s16
v_readfirstlane_b32 s18, v0
s_mul_i32 s17, s17, s18
s_mul_hi_u32 s17, s18, s17
s_add_i32 s18, s18, s17
s_mul_hi_u32 s17, s14, s18
s_mul_i32 s18, s17, s16
s_sub_i32 s14, s14, s18
s_add_i32 s18, s17, 1
s_sub_i32 s19, s14, s16
s_cmp_ge_u32 s14, s16
s_cselect_b32 s17, s18, s17
s_cselect_b32 s14, s19, s14
s_add_i32 s18, s17, 1
s_cmp_ge_u32 s14, s16
s_cselect_b32 s14, s18, s17
s_xor_b32 s14, s14, s9
s_sub_i32 s9, s14, s9
s_mul_i32 s0, s9, s0
s_sub_i32 s54, s8, s0
s_add_i32 s54, s54, s15
v_lshlrev_b32_e32 v0, 1, v34
v_and_b32_e32 v0, 0x70, v0
v_or_b32_e32 v1, s1, v0
v_or_b32_e32 v4, 4, v1
v_or_b32_e32 v6, 8, v1
v_or_b32_e32 v8, 12, v1
v_or_b32_e32 v2, 0x80, v1
v_or_b32_e32 v3, 0x84, v1
v_or_b32_e32 v5, 0x88, v1
v_or_b32_e32 v7, 0x8c, v1
v_lshlrev_b32_e32 v35, 3, v34
v_and_b32_e32 v0, 56, v35
s_lshl_b32 s8, s54, 8
s_mul_i32 s14, s8, s11
s_ashr_i32 s15, s14, 31
s_lshl_b64 s[14:15], s[14:15], 1
s_add_u32 s16, s2, s14
s_addc_u32 s35, s3, s15
s_lshl_b32 s0, s9, 8
s_mul_i32 s14, s0, s12
s_ashr_i32 s15, s14, 31
s_lshl_b64 s[14:15], s[14:15], 1
s_add_u32 s20, s4, s14
s_addc_u32 s9, s5, s15
v_mul_lo_u32 v10, v1, s11
v_mul_lo_u32 v11, v4, s11
v_mul_lo_u32 v12, v6, s11
v_mul_lo_u32 v13, v8, s11
v_mul_lo_u32 v14, v2, s11
v_mul_lo_u32 v15, v3, s11
v_mul_lo_u32 v16, v5, s11
v_mul_lo_u32 v17, v7, s11
v_mad_u64_u32 v[2:3], s[4:5], v1, s12, v[0:1]
v_mad_u64_u32 v[4:5], s[4:5], v4, s12, v[0:1]
v_mad_u64_u32 v[6:7], s[4:5], v6, s12, v[0:1]
v_mad_u64_u32 v[8:9], s[4:5], v8, s12, v[0:1]
s_lshl_b32 s4, s12, 8
s_ashr_i32 s5, s4, 1
s_add_i32 s56, s10, 63
s_ashr_i32 s4, s56, 31
s_lshr_b32 s4, s4, 26
s_add_i32 s4, s56, s4
s_ashr_i32 s55, s4, 6
s_and_b32 s17, s35, 0xffff
s_mov_b32 s19, 0x27000
s_mov_b32 s18, 0x7ffffffe
s_mul_i32 s14, s1, 0x420
s_add_i32 s4, s14, 0
v_add_lshl_u32 v254, v10, v0, 1
s_mov_b32 m0, s4
s_nop 0
buffer_load_dwordx4 v254, s[16:19], 0, offen, lds
s_add_i32 s15, s14, 0x1080
s_add_i32 s10, s4, 0x1080
v_add_lshl_u32 v7, v11, v0, 1
s_mov_b32 m0, s10
s_nop 0
buffer_load_dwordx4 v7, s[16:19], 0, offen, lds
s_add_i32 s51, s14, 0x2100
s_add_i32 s12, s4, 0x2100
v_add_lshl_u32 v9, v12, v0, 1
s_mov_b32 m0, s12
s_nop 0
buffer_load_dwordx4 v9, s[16:19], 0, offen, lds
s_add_i32 s52, s14, 0x3180
s_add_i32 s25, s4, 0x3180
v_add_lshl_u32 v10, v13, v0, 1
s_mov_b32 m0, s25
s_nop 0
buffer_load_dwordx4 v10, s[16:19], 0, offen, lds
s_add_i32 s26, s4, 0x4200
v_add_lshl_u32 v11, v14, v0, 1
s_mov_b32 m0, s26
s_nop 0
buffer_load_dwordx4 v11, s[16:19], 0, offen, lds
s_add_i32 s27, s4, 0x5280
v_add_lshl_u32 v12, v15, v0, 1
s_mov_b32 m0, s27
s_nop 0
buffer_load_dwordx4 v12, s[16:19], 0, offen, lds
s_add_i32 s28, s4, 0x6300
v_add_lshl_u32 v13, v16, v0, 1
s_mov_b32 m0, s28
s_nop 0
buffer_load_dwordx4 v13, s[16:19], 0, offen, lds
s_add_i32 s29, s4, 0x7380
v_add_lshl_u32 v14, v17, v0, 1
s_mov_b32 m0, s29
s_nop 0
buffer_load_dwordx4 v14, s[16:19], 0, offen, lds
s_and_b32 s21, s9, 0xffff
s_mov_b32 s22, s18
s_mov_b32 s23, s19
s_add_i32 s57, 0, 0x107e0
s_add_i32 s30, s57, s14
v_lshlrev_b32_e32 v0, 1, v2
s_mov_b32 m0, s30
s_nop 0
buffer_load_dwordx4 v0, s[20:23], 0, offen, lds
s_add_i32 s31, s57, s15
v_lshlrev_b32_e32 v1, 1, v4
s_mov_b32 m0, s31
s_nop 0
buffer_load_dwordx4 v1, s[20:23], 0, offen, lds
s_add_i32 s33, s57, s51
v_lshlrev_b32_e32 v3, 1, v6
s_mov_b32 m0, s33
s_nop 0
buffer_load_dwordx4 v3, s[20:23], 0, offen, lds
s_add_i32 s34, s57, s52
v_lshlrev_b32_e32 v5, 1, v8
s_mov_b32 m0, s34
s_nop 0
buffer_load_dwordx4 v5, s[20:23], 0, offen, lds
s_add_u32 s16, s16, 0x80
s_addc_u32 s17, s35, 0
s_add_i32 s38, 0, 0x18bc0
s_add_i32 s35, s38, s14
v_add_lshl_u32 v15, v2, s5, 1
s_mov_b32 m0, s35
s_nop 0
buffer_load_dwordx4 v15, s[20:23], 0, offen, lds
s_add_i32 s36, s38, s15
v_add_lshl_u32 v22, v4, s5, 1
s_mov_b32 m0, s36
s_nop 0
buffer_load_dwordx4 v22, s[20:23], 0, offen, lds
s_add_i32 s37, s38, s51
v_add_lshl_u32 v23, v6, s5, 1
s_mov_b32 m0, s37
s_nop 0
buffer_load_dwordx4 v23, s[20:23], 0, offen, lds
s_add_i32 s38, s38, s52
v_add_lshl_u32 v6, v8, s5, 1
s_mov_b32 m0, s38
s_nop 0
buffer_load_dwordx4 v6, s[20:23], 0, offen, lds
s_waitcnt lgkmcnt(0)
s_barrier
s_and_b32 s17, s17, 0xffff
s_add_i32 s39, s35, 0xfffef840
s_mov_b32 m0, s39
s_nop 0
buffer_load_dwordx4 v254, s[16:19], 0, offen, lds
s_add_i32 s40, s35, 0xffff08c0
s_mov_b32 m0, s40
s_nop 0
buffer_load_dwordx4 v7, s[16:19], 0, offen, lds
s_add_i32 s41, s35, 0xffff1940
s_mov_b32 m0, s41
s_nop 0
buffer_load_dwordx4 v9, s[16:19], 0, offen, lds
s_add_i32 s42, s35, 0xffff29c0
s_mov_b32 m0, s42
s_nop 0
buffer_load_dwordx4 v10, s[16:19], 0, offen, lds
s_add_i32 s43, s35, 0xffff3a40
s_mov_b32 m0, s43
s_nop 0
buffer_load_dwordx4 v11, s[16:19], 0, offen, lds
s_add_i32 s44, s35, 0xffff4ac0
s_mov_b32 m0, s44
s_nop 0
buffer_load_dwordx4 v12, s[16:19], 0, offen, lds
s_add_i32 s45, s35, 0xffff5b40
s_mov_b32 m0, s45
s_nop 0
buffer_load_dwordx4 v13, s[16:19], 0, offen, lds
s_add_i32 s46, s35, 0xffff6bc0
s_mov_b32 m0, s46
s_nop 0
buffer_load_dwordx4 v14, s[16:19], 0, offen, lds
s_add_i32 s50, 0, 0x149e0
s_add_i32 s47, s50, s14
v_add_u32_e32 v8, 0x80, v0
s_mov_b32 m0, s47
s_nop 0
buffer_load_dwordx4 v8, s[20:23], 0, offen, lds
s_add_i32 s48, s50, s15
v_add_u32_e32 v16, 0x80, v1
s_mov_b32 m0, s48
s_nop 0
buffer_load_dwordx4 v16, s[20:23], 0, offen, lds
s_add_i32 s49, s50, s51
v_add_u32_e32 v17, 0x80, v3
s_mov_b32 m0, s49
s_nop 0
buffer_load_dwordx4 v17, s[20:23], 0, offen, lds
s_add_i32 s50, s50, s52
v_add_u32_e32 v5, 0x80, v5
s_mov_b32 m0, s50
s_nop 0
buffer_load_dwordx4 v5, s[20:23], 0, offen, lds
s_add_u32 s20, s20, 0x80
s_addc_u32 s21, s9, 0
s_and_b32 s17, s21, 0xffff
s_mov_b32 s16, s20
s_add_i32 s5, 0, 0x1cdc0
s_add_i32 s22, s5, s14
s_mov_b32 m0, s22
s_nop 0
buffer_load_dwordx4 v15, s[16:19], 0, offen, lds
s_add_i32 s23, s5, s15
s_mov_b32 m0, s23
s_nop 0
buffer_load_dwordx4 v22, s[16:19], 0, offen, lds
s_add_i32 s51, s5, s51
s_mov_b32 m0, s51
s_nop 0
buffer_load_dwordx4 v23, s[16:19], 0, offen, lds
s_add_i32 s52, s5, s52
s_mov_b32 m0, s52
s_nop 0
buffer_load_dwordx4 v6, s[16:19], 0, offen, lds
s_waitcnt vmcnt(20), lgkmcnt(0)
s_barrier
v_and_b32_e32 v4, 15, v34
v_lshlrev_b32_e32 v0, 10, v4
v_and_b32_e32 v1, 48, v34
s_cmp_lt_u32 s1, 2
s_cselect_b64 s[14:15], -1, 0
s_and_b64 s[16:17], s[14:15], exec
s_cselect_b32 s5, 0, 0x100
v_or3_b32 v2, v1, s5, v0
v_lshlrev_b32_e32 v3, 5, v4
v_add_u32_e32 v2, v2, v3
v_add_u32_e32 v255, 0, v2
ds_read_b128 a[56:59], v255
ds_read_b128 a[60:63], v255, offset:64
ds_read_b128 a[48:51], v255, offset:128
ds_read_b128 a[52:55], v255, offset:192
ds_read_b128 a[40:43], v255, offset:512
ds_read_b128 a[44:47], v255, offset:576
ds_read_b128 a[32:35], v255, offset:640
ds_read_b128 a[36:39], v255, offset:704
ds_read_b128 a[24:27], v255, offset:16896
ds_read_b128 a[28:31], v255, offset:16960
ds_read_b128 a[16:19], v255, offset:17024
ds_read_b128 a[20:23], v255, offset:17088
ds_read_b128 a[8:11], v255, offset:17408
ds_read_b128 a[12:15], v255, offset:17472
ds_read_b128 a[0:3], v255, offset:17536
ds_read_b128 a[4:7], v255, offset:17600
s_and_b32 s5, s53, 64
v_or_b32_e32 v0, v0, v1
v_add_u32_e32 v0, v0, v3
v_lshl_add_u32 v36, s5, 2, v0
v_add_u32_e32 v0, s57, v36
ds_read_b128 a[88:91], v0
ds_read_b128 a[92:95], v0, offset:64
ds_read_b128 a[80:83], v0, offset:128
ds_read_b128 a[84:87], v0, offset:192
ds_read_b128 a[72:75], v0, offset:512
ds_read_b128 a[76:79], v0, offset:576
ds_read_b128 a[64:67], v0, offset:640
ds_read_b128 a[68:71], v0, offset:704
s_add_i32 s9, s55, -1
s_cmpk_lt_i32 s56, 0x80
s_cbranch_scc1 .LBB0_4
; %bb.1:
v_accvgpr_write_b32 a154, v4
v_accvgpr_write_b32 a153, v35
v_accvgpr_write_b32 a152, v34
s_add_i32 s53, s55, -2
s_mul_i32 s11, s11, s54
s_lshl_b32 s16, s11, 8
s_ashr_i32 s17, s16, 31
s_lshl_b64 s[16:17], s[16:17], 1
s_add_u32 s2, s2, s16
s_addc_u32 s3, s3, s17
s_add_u32 s2, s2, 0x180
s_addc_u32 s3, s3, 0
v_mov_b32_e32 v126, 0
v_accvgpr_write_b32 a155, v36
v_add_u32_e32 v0, 0, v36
v_add_u32_e32 v1, 0x18bc0, v0
v_accvgpr_write_b32 a156, v1
v_add_u32_e32 v1, 0x149e0, v0
v_accvgpr_write_b32 a158, v1
v_add_u32_e32 v1, 0x1cdc0, v0
v_accvgpr_write_b32 a159, v1
v_add_u32_e32 v0, 0x107e0, v0
v_accvgpr_write_b32 a160, v0
v_mov_b32_e32 v127, v126
v_mov_b32_e32 v128, v126
v_mov_b32_e32 v129, v126
v_mov_b32_e32 v122, v126
v_mov_b32_e32 v123, v126
v_mov_b32_e32 v124, v126
v_mov_b32_e32 v125, v126
v_mov_b32_e32 v118, v126
v_mov_b32_e32 v119, v126
v_mov_b32_e32 v120, v126
v_mov_b32_e32 v121, v126
v_mov_b32_e32 v114, v126
v_mov_b32_e32 v115, v126
v_mov_b32_e32 v116, v126
v_mov_b32_e32 v117, v126
v_mov_b32_e32 v110, v126
v_mov_b32_e32 v111, v126
v_mov_b32_e32 v112, v126
v_mov_b32_e32 v113, v126
v_mov_b32_e32 v0, v126
v_mov_b32_e32 v1, v126
v_mov_b32_e32 v2, v126
v_mov_b32_e32 v3, v126
v_accvgpr_write_b32 a165, v3
v_accvgpr_write_b32 a164, v2
v_accvgpr_write_b32 a163, v1
v_accvgpr_write_b32 a162, v0
v_accvgpr_write_b32 a169, v3
v_accvgpr_write_b32 a168, v2
v_accvgpr_write_b32 a167, v1
v_accvgpr_write_b32 a166, v0
v_accvgpr_write_b32 a214, v126
v_accvgpr_write_b32 a215, v126
v_accvgpr_write_b32 a216, v126
v_accvgpr_write_b32 a217, v126
v_accvgpr_write_b32 a210, v126
v_accvgpr_write_b32 a211, v126
v_accvgpr_write_b32 a212, v126
v_accvgpr_write_b32 a213, v126
v_mov_b32_e32 v246, v126
v_mov_b32_e32 v247, v126
v_mov_b32_e32 v248, v126
v_mov_b32_e32 v249, v126
v_mov_b32_e32 v86, v126
v_mov_b32_e32 v87, v126
v_mov_b32_e32 v88, v126
v_mov_b32_e32 v89, v126
v_mov_b32_e32 v82, v126
v_mov_b32_e32 v83, v126
v_mov_b32_e32 v84, v126
v_mov_b32_e32 v85, v126
v_mov_b32_e32 v78, v126
v_mov_b32_e32 v79, v126
v_mov_b32_e32 v80, v126
v_mov_b32_e32 v81, v126
v_mov_b32_e32 v74, v126
v_mov_b32_e32 v75, v126
v_mov_b32_e32 v76, v126
v_mov_b32_e32 v77, v126
v_mov_b32_e32 v70, v126
v_mov_b32_e32 v71, v126
v_mov_b32_e32 v72, v126
v_mov_b32_e32 v73, v126
v_mov_b32_e32 v66, v126
v_mov_b32_e32 v67, v126
v_mov_b32_e32 v68, v126
v_mov_b32_e32 v69, v126
v_mov_b32_e32 v62, v126
v_mov_b32_e32 v63, v126
v_mov_b32_e32 v64, v126
v_mov_b32_e32 v65, v126
v_mov_b32_e32 v58, v126
v_mov_b32_e32 v59, v126
v_mov_b32_e32 v60, v126
v_mov_b32_e32 v61, v126
v_accvgpr_write_b32 a173, v3
v_accvgpr_write_b32 a172, v2
v_accvgpr_write_b32 a171, v1
v_accvgpr_write_b32 a170, v0
v_accvgpr_write_b32 a177, v3
v_accvgpr_write_b32 a176, v2
v_accvgpr_write_b32 a175, v1
v_accvgpr_write_b32 a174, v0
v_mov_b32_e32 v98, v126
v_mov_b32_e32 v99, v126
v_mov_b32_e32 v100, v126
v_mov_b32_e32 v101, v126
v_mov_b32_e32 v94, v126
v_mov_b32_e32 v95, v126
v_mov_b32_e32 v96, v126
v_mov_b32_e32 v97, v126
v_mov_b32_e32 v242, v126
v_mov_b32_e32 v243, v126
v_mov_b32_e32 v244, v126
v_mov_b32_e32 v245, v126
v_accvgpr_write_b32 a124, v126
v_accvgpr_write_b32 a125, v126
v_accvgpr_write_b32 a126, v126
v_accvgpr_write_b32 a127, v126
v_accvgpr_write_b32 a120, v126
v_accvgpr_write_b32 a121, v126
v_accvgpr_write_b32 a122, v126
v_accvgpr_write_b32 a123, v126
v_accvgpr_write_b32 a250, v126
v_accvgpr_write_b32 a251, v126
v_accvgpr_write_b32 a252, v126
v_accvgpr_write_b32 a253, v126
v_accvgpr_write_b32 a246, v126
v_accvgpr_write_b32 a247, v126
v_accvgpr_write_b32 a248, v126
v_accvgpr_write_b32 a249, v126
v_accvgpr_write_b32 a242, v126
v_accvgpr_write_b32 a243, v126
v_accvgpr_write_b32 a244, v126
v_accvgpr_write_b32 a245, v126
v_accvgpr_write_b32 a206, v126
v_accvgpr_write_b32 a207, v126
v_accvgpr_write_b32 a208, v126
v_accvgpr_write_b32 a209, v126
v_accvgpr_write_b32 a161, v14
v_accvgpr_write_b32 a222, v126
v_accvgpr_write_b32 a178, v15
v_accvgpr_write_b32 a223, v126
v_accvgpr_write_b32 a179, v16
v_accvgpr_write_b32 a224, v126
v_accvgpr_write_b32 a180, v17
v_accvgpr_write_b32 a225, v126
v_accvgpr_write_b32 a181, v10
v_accvgpr_write_b32 a218, v126
v_accvgpr_write_b32 a182, v11
v_accvgpr_write_b32 a219, v126
v_accvgpr_write_b32 a183, v12
v_accvgpr_write_b32 a220, v126
v_accvgpr_write_b32 a184, v13
v_accvgpr_write_b32 a221, v126
v_mov_b32_e32 v250, v126
v_mov_b32_e32 v251, v126
v_mov_b32_e32 v252, v126
v_mov_b32_e32 v253, v126
v_accvgpr_write_b32 a96, v126
v_accvgpr_write_b32 a97, v126
v_accvgpr_write_b32 a98, v126
v_accvgpr_write_b32 a99, v126
v_mov_b32_e32 v30, v126
v_mov_b32_e32 v31, v126
v_mov_b32_e32 v32, v126
v_mov_b32_e32 v33, v126
v_mov_b32_e32 v130, v126
v_mov_b32_e32 v131, v126
v_mov_b32_e32 v132, v126
v_mov_b32_e32 v133, v126
v_accvgpr_write_b32 a100, v126
v_accvgpr_write_b32 a101, v126
v_accvgpr_write_b32 a102, v126
v_accvgpr_write_b32 a103, v126
v_mov_b32_e32 v24, v126
v_mov_b32_e32 v25, v126
v_mov_b32_e32 v26, v126
v_mov_b32_e32 v27, v126
v_accvgpr_write_b32 a189, v27
v_accvgpr_write_b32 a188, v26
v_accvgpr_write_b32 a187, v25
v_accvgpr_write_b32 a186, v24
v_accvgpr_write_b32 a193, v27
v_accvgpr_write_b32 a192, v26
v_accvgpr_write_b32 a191, v25
v_accvgpr_write_b32 a190, v24
v_mov_b32_e32 v230, v126
v_mov_b32_e32 v231, v126
v_mov_b32_e32 v232, v126
v_mov_b32_e32 v233, v126
v_mov_b32_e32 v226, v126
v_mov_b32_e32 v227, v126
v_mov_b32_e32 v228, v126
v_mov_b32_e32 v229, v126
v_mov_b32_e32 v222, v126
v_mov_b32_e32 v223, v126
v_mov_b32_e32 v224, v126
v_mov_b32_e32 v225, v126
v_mov_b32_e32 v218, v126
v_mov_b32_e32 v219, v126
v_mov_b32_e32 v220, v126
v_mov_b32_e32 v221, v126
v_mov_b32_e32 v214, v126
v_mov_b32_e32 v215, v126
v_mov_b32_e32 v216, v126
v_mov_b32_e32 v217, v126
v_mov_b32_e32 v210, v126
v_mov_b32_e32 v211, v126
v_mov_b32_e32 v212, v126
v_mov_b32_e32 v213, v126
v_mov_b32_e32 v206, v126
v_mov_b32_e32 v207, v126
v_mov_b32_e32 v208, v126
v_mov_b32_e32 v209, v126
v_mov_b32_e32 v202, v126
v_mov_b32_e32 v203, v126
v_mov_b32_e32 v204, v126
v_mov_b32_e32 v205, v126
v_mov_b32_e32 v198, v126
v_mov_b32_e32 v199, v126
v_mov_b32_e32 v200, v126
v_mov_b32_e32 v201, v126
v_mov_b32_e32 v194, v126
v_mov_b32_e32 v195, v126
v_mov_b32_e32 v196, v126
v_mov_b32_e32 v197, v126
v_mov_b32_e32 v190, v126
v_mov_b32_e32 v191, v126
v_mov_b32_e32 v192, v126
v_mov_b32_e32 v193, v126
v_mov_b32_e32 v186, v126
v_mov_b32_e32 v187, v126
v_mov_b32_e32 v188, v126
v_mov_b32_e32 v189, v126
v_mov_b32_e32 v182, v126
v_mov_b32_e32 v183, v126
v_mov_b32_e32 v184, v126
v_mov_b32_e32 v185, v126
v_mov_b32_e32 v178, v126
v_mov_b32_e32 v179, v126
v_mov_b32_e32 v180, v126
v_mov_b32_e32 v181, v126
v_mov_b32_e32 v174, v126
v_mov_b32_e32 v175, v126
v_mov_b32_e32 v176, v126
v_mov_b32_e32 v177, v126
v_mov_b32_e32 v170, v126
v_mov_b32_e32 v171, v126
v_mov_b32_e32 v172, v126
v_mov_b32_e32 v173, v126
v_mov_b32_e32 v166, v126
v_mov_b32_e32 v167, v126
v_mov_b32_e32 v168, v126
v_mov_b32_e32 v169, v126
v_mov_b32_e32 v162, v126
v_mov_b32_e32 v163, v126
v_mov_b32_e32 v164, v126
v_mov_b32_e32 v165, v126
v_mov_b32_e32 v158, v126
v_mov_b32_e32 v159, v126
v_mov_b32_e32 v160, v126
v_mov_b32_e32 v161, v126
v_mov_b32_e32 v154, v126
v_mov_b32_e32 v155, v126
v_mov_b32_e32 v156, v126
v_mov_b32_e32 v157, v126
v_mov_b32_e32 v150, v126
v_mov_b32_e32 v151, v126
v_mov_b32_e32 v152, v126
v_mov_b32_e32 v153, v126
v_mov_b32_e32 v146, v126
v_mov_b32_e32 v147, v126
v_mov_b32_e32 v148, v126
v_mov_b32_e32 v149, v126
v_mov_b32_e32 v142, v126
v_mov_b32_e32 v143, v126
v_mov_b32_e32 v144, v126
v_mov_b32_e32 v145, v126
v_mov_b32_e32 v138, v126
v_mov_b32_e32 v139, v126
v_mov_b32_e32 v140, v126
v_mov_b32_e32 v141, v126
v_mov_b32_e32 v134, v126
v_mov_b32_e32 v135, v126
v_mov_b32_e32 v136, v126
v_mov_b32_e32 v137, v126
v_accvgpr_write_b32 a185, v6
v_mov_b32_e32 v6, v126
v_accvgpr_write_b32 a194, v7
v_mov_b32_e32 v7, v126
v_accvgpr_write_b32 a195, v8
v_mov_b32_e32 v8, v126
v_accvgpr_write_b32 a196, v9
v_mov_b32_e32 v9, v126
v_accvgpr_write_b32 a197, v5
v_bfrev_b32_e32 v21, 1
v_accvgpr_read_b32 v35, a156
v_accvgpr_read_b32 v37, a194
v_accvgpr_read_b32 v42, a196
v_accvgpr_read_b32 v40, a181
v_accvgpr_read_b32 v45, a182
v_accvgpr_read_b32 v0, a183
v_accvgpr_read_b32 v36, a184
v_accvgpr_read_b32 v41, a161
v_accvgpr_read_b32 v38, a195
v_accvgpr_read_b32 v43, a179
v_accvgpr_read_b32 v34, a180
v_accvgpr_read_b32 v39, a197
v_accvgpr_read_b32 v46, a158
v_accvgpr_read_b32 v1, a178
v_accvgpr_read_b32 v44, a185
v_accvgpr_read_b32 v2, a159
s_mov_b32 s58, s18
s_mov_b32 s59, s19
v_accvgpr_read_b32 v3, a160
.LBB0_2:
s_add_u32 s16, s2, 0xffffff80
s_addc_u32 s11, s3, -1
s_waitcnt lgkmcnt(0)
v_mfma_f32_16x16x32_f16 a[96:99], a[88:91], a[56:59], a[96:99]
v_mfma_f32_16x16x32_f16 a[96:99], a[92:95], a[60:63], a[96:99]
v_mfma_f32_16x16x32_f16 v[30:33], a[80:83], a[56:59], v[30:33]
v_mfma_f32_16x16x32_f16 v[30:33], a[84:87], a[60:63], v[30:33]
s_waitcnt vmcnt(16), lgkmcnt(0)
s_barrier
ds_read_b128 a[104:107], v35
v_mfma_f32_16x16x32_f16 v[130:133], a[72:75], a[56:59], v[130:133]
ds_read_b128 a[108:111], v35, offset:64
v_mfma_f32_16x16x32_f16 v[130:133], a[76:79], a[60:63], v[130:133]
ds_read_b128 a[112:115], v35, offset:128
v_mfma_f32_16x16x32_f16 a[100:103], a[64:67], a[56:59], a[100:103]
ds_read_b128 a[116:119], v35, offset:192
v_mfma_f32_16x16x32_f16 a[100:103], a[68:71], a[60:63], a[100:103]
ds_read_b128 a[128:131], v35, offset:512
v_mfma_f32_16x16x32_f16 a[186:189], a[88:91], a[48:51], a[186:189]
ds_read_b128 a[132:135], v35, offset:576
v_mfma_f32_16x16x32_f16 a[186:189], a[92:95], a[52:55], a[186:189]
ds_read_b128 a[136:139], v35, offset:640
v_mfma_f32_16x16x32_f16 a[190:193], a[80:83], a[48:51], a[190:193]
ds_read_b128 a[140:143], v35, offset:704
v_mfma_f32_16x16x32_f16 a[190:193], a[84:87], a[52:55], a[190:193]
v_mfma_f32_16x16x32_f16 v[230:233], a[72:75], a[48:51], v[230:233]
v_mfma_f32_16x16x32_f16 v[230:233], a[76:79], a[52:55], v[230:233]
s_cmp_eq_u32 s53, s24
s_cselect_b64 vcc, -1, 0
s_and_b32 s17, s11, 0xffff
v_cndmask_b32_e32 v20, v254, v21, vcc
s_mov_b32 m0, s4
v_mfma_f32_16x16x32_f16 v[226:229], a[64:67], a[48:51], v[226:229]
buffer_load_dwordx4 v20, s[16:19], 0, offen, lds
v_mfma_f32_16x16x32_f16 v[226:229], a[68:71], a[52:55], v[226:229]
v_mfma_f32_16x16x32_f16 v[222:225], a[88:91], a[40:43], v[222:225]
v_mfma_f32_16x16x32_f16 v[222:225], a[92:95], a[44:47], v[222:225]
v_accvgpr_write_b32 a157, v254
v_cndmask_b32_e32 v37, v37, v21, vcc
s_mov_b32 m0, s10
v_mfma_f32_16x16x32_f16 v[218:221], a[80:83], a[40:43], v[218:221]
buffer_load_dwordx4 v37, s[16:19], 0, offen, lds
v_mfma_f32_16x16x32_f16 v[218:221], a[84:87], a[44:47], v[218:221]
v_mfma_f32_16x16x32_f16 v[214:217], a[72:75], a[40:43], v[214:217]
v_mfma_f32_16x16x32_f16 v[214:217], a[76:79], a[44:47], v[214:217]
v_cndmask_b32_e32 v42, v42, v21, vcc
s_mov_b32 m0, s12
v_mfma_f32_16x16x32_f16 v[210:213], a[64:67], a[40:43], v[210:213]
buffer_load_dwordx4 v42, s[16:19], 0, offen, lds
v_mfma_f32_16x16x32_f16 v[210:213], a[68:71], a[44:47], v[210:213]
v_mfma_f32_16x16x32_f16 v[206:209], a[88:91], a[32:35], v[206:209]
v_mfma_f32_16x16x32_f16 v[206:209], a[92:95], a[36:39], v[206:209]
v_cndmask_b32_e32 v40, v40, v21, vcc
s_mov_b32 m0, s25
v_mfma_f32_16x16x32_f16 v[202:205], a[80:83], a[32:35], v[202:205]
buffer_load_dwordx4 v40, s[16:19], 0, offen, lds
v_mfma_f32_16x16x32_f16 v[202:205], a[84:87], a[36:39], v[202:205]
v_mfma_f32_16x16x32_f16 v[198:201], a[72:75], a[32:35], v[198:201]
v_mfma_f32_16x16x32_f16 v[198:201], a[76:79], a[36:39], v[198:201]
v_cndmask_b32_e32 v45, v45, v21, vcc
s_mov_b32 m0, s26
v_mfma_f32_16x16x32_f16 v[194:197], a[64:67], a[32:35], v[194:197]
buffer_load_dwordx4 v45, s[16:19], 0, offen, lds
v_mfma_f32_16x16x32_f16 v[194:197], a[68:71], a[36:39], v[194:197]
v_mfma_f32_16x16x32_f16 v[190:193], a[88:91], a[24:27], v[190:193]
v_mfma_f32_16x16x32_f16 v[190:193], a[92:95], a[28:31], v[190:193]
v_cndmask_b32_e32 v0, v0, v21, vcc
s_mov_b32 m0, s27
v_mfma_f32_16x16x32_f16 v[186:189], a[80:83], a[24:27], v[186:189]
buffer_load_dwordx4 v0, s[16:19], 0, offen, lds
v_mfma_f32_16x16x32_f16 v[186:189], a[84:87], a[28:31], v[186:189]
v_mfma_f32_16x16x32_f16 v[182:185], a[72:75], a[24:27], v[182:185]
v_mfma_f32_16x16x32_f16 v[182:185], a[76:79], a[28:31], v[182:185]
v_cndmask_b32_e32 v36, v36, v21, vcc
s_mov_b32 m0, s28
v_mfma_f32_16x16x32_f16 v[178:181], a[64:67], a[24:27], v[178:181]
buffer_load_dwordx4 v36, s[16:19], 0, offen, lds
v_mfma_f32_16x16x32_f16 v[178:181], a[68:71], a[28:31], v[178:181]
v_mfma_f32_16x16x32_f16 v[174:177], a[88:91], a[16:19], v[174:177]
v_mfma_f32_16x16x32_f16 v[174:177], a[92:95], a[20:23], v[174:177]
v_cndmask_b32_e32 v41, v41, v21, vcc
s_mov_b32 m0, s29
v_mfma_f32_16x16x32_f16 v[170:173], a[80:83], a[16:19], v[170:173]
buffer_load_dwordx4 v41, s[16:19], 0, offen, lds
v_mfma_f32_16x16x32_f16 v[170:173], a[84:87], a[20:23], v[170:173]
v_mfma_f32_16x16x32_f16 v[166:169], a[72:75], a[16:19], v[166:169]
v_mfma_f32_16x16x32_f16 v[166:169], a[76:79], a[20:23], v[166:169]
s_and_b32 s17, s21, 0xffff
s_mov_b32 s16, s20
v_cndmask_b32_e32 v38, v38, v21, vcc
s_mov_b32 m0, s30
v_mfma_f32_16x16x32_f16 v[162:165], a[64:67], a[16:19], v[162:165]
buffer_load_dwordx4 v38, s[16:19], 0, offen, lds
v_mfma_f32_16x16x32_f16 v[162:165], a[68:71], a[20:23], v[162:165]
v_mfma_f32_16x16x32_f16 v[158:161], a[88:91], a[8:11], v[158:161]
v_mfma_f32_16x16x32_f16 v[158:161], a[92:95], a[12:15], v[158:161]
v_cndmask_b32_e32 v43, v43, v21, vcc
s_mov_b32 m0, s31
v_mfma_f32_16x16x32_f16 v[154:157], a[80:83], a[8:11], v[154:157]
buffer_load_dwordx4 v43, s[16:19], 0, offen, lds
v_mfma_f32_16x16x32_f16 v[154:157], a[84:87], a[12:15], v[154:157]
v_mfma_f32_16x16x32_f16 v[150:153], a[72:75], a[8:11], v[150:153]
v_mfma_f32_16x16x32_f16 v[150:153], a[76:79], a[12:15], v[150:153]
v_cndmask_b32_e32 v34, v34, v21, vcc
s_mov_b32 m0, s33
v_mfma_f32_16x16x32_f16 v[146:149], a[64:67], a[8:11], v[146:149]
buffer_load_dwordx4 v34, s[16:19], 0, offen, lds
v_mfma_f32_16x16x32_f16 v[146:149], a[68:71], a[12:15], v[146:149]
v_mfma_f32_16x16x32_f16 v[142:145], a[88:91], a[0:3], v[142:145]
v_mfma_f32_16x16x32_f16 v[142:145], a[92:95], a[4:7], v[142:145]
v_cndmask_b32_e32 v39, v39, v21, vcc
s_mov_b32 m0, s34
v_mfma_f32_16x16x32_f16 v[138:141], a[80:83], a[0:3], v[138:141]
buffer_load_dwordx4 v39, s[16:19], 0, offen, lds
v_mfma_f32_16x16x32_f16 v[138:141], a[84:87], a[4:7], v[138:141]
v_mfma_f32_16x16x32_f16 v[134:137], a[72:75], a[0:3], v[134:137]
v_mfma_f32_16x16x32_f16 v[134:137], a[76:79], a[4:7], v[134:137]
v_mfma_f32_16x16x32_f16 v[6:9], a[64:67], a[0:3], v[6:9]
v_mfma_f32_16x16x32_f16 v[6:9], a[68:71], a[4:7], v[6:9]
s_waitcnt lgkmcnt(0)
v_mfma_f32_16x16x32_f16 v[126:129], a[104:107], a[56:59], v[126:129]
v_mfma_f32_16x16x32_f16 v[126:129], a[108:111], a[60:63], v[126:129]
v_mfma_f32_16x16x32_f16 v[122:125], a[112:115], a[56:59], v[122:125]
v_mfma_f32_16x16x32_f16 v[122:125], a[116:119], a[60:63], v[122:125]
v_mfma_f32_16x16x32_f16 v[118:121], a[128:131], a[56:59], v[118:121]
v_mfma_f32_16x16x32_f16 v[118:121], a[132:135], a[60:63], v[118:121]
v_mfma_f32_16x16x32_f16 v[114:117], a[136:139], a[56:59], v[114:117]
v_mfma_f32_16x16x32_f16 v[114:117], a[140:143], a[60:63], v[114:117]
v_mfma_f32_16x16x32_f16 v[110:113], a[104:107], a[48:51], v[110:113]
v_mfma_f32_16x16x32_f16 v[110:113], a[108:111], a[52:55], v[110:113]
v_mfma_f32_16x16x32_f16 a[162:165], a[112:115], a[48:51], a[162:165]
v_mfma_f32_16x16x32_f16 a[162:165], a[116:119], a[52:55], a[162:165]
v_mfma_f32_16x16x32_f16 a[166:169], a[128:131], a[48:51], a[166:169]
v_mfma_f32_16x16x32_f16 a[166:169], a[132:135], a[52:55], a[166:169]
v_mfma_f32_16x16x32_f16 a[214:217], a[136:139], a[48:51], a[214:217]
v_mfma_f32_16x16x32_f16 a[214:217], a[140:143], a[52:55], a[214:217]
v_mfma_f32_16x16x32_f16 a[210:213], a[104:107], a[40:43], a[210:213]
v_mfma_f32_16x16x32_f16 a[210:213], a[108:111], a[44:47], a[210:213]
v_mfma_f32_16x16x32_f16 v[246:249], a[112:115], a[40:43], v[246:249]
v_mfma_f32_16x16x32_f16 v[246:249], a[116:119], a[44:47], v[246:249]
s_waitcnt vmcnt(16), lgkmcnt(0)
s_barrier
ds_read_b128 a[144:147], v255, offset:33792
v_mfma_f32_16x16x32_f16 v[86:89], a[128:131], a[40:43], v[86:89]
ds_read_b128 a[148:151], v255, offset:33856
v_mfma_f32_16x16x32_f16 v[86:89], a[132:135], a[44:47], v[86:89]
ds_read_b128 a[158:161], v255, offset:33920
v_mfma_f32_16x16x32_f16 v[82:85], a[136:139], a[40:43], v[82:85]
ds_read_b128 a[178:181], v255, offset:33984
v_mfma_f32_16x16x32_f16 v[82:85], a[140:143], a[44:47], v[82:85]
ds_read_b128 a[182:185], v255, offset:34304
v_mfma_f32_16x16x32_f16 v[78:81], a[104:107], a[32:35], v[78:81]
ds_read_b128 a[194:197], v255, offset:34368
v_mfma_f32_16x16x32_f16 v[78:81], a[108:111], a[36:39], v[78:81]
ds_read_b128 a[198:201], v255, offset:34432
v_mfma_f32_16x16x32_f16 v[74:77], a[112:115], a[32:35], v[74:77]
ds_read_b128 a[202:205], v255, offset:34496
v_mfma_f32_16x16x32_f16 v[74:77], a[116:119], a[36:39], v[74:77]
ds_read_b128 a[226:229], v255, offset:50688
v_mfma_f32_16x16x32_f16 v[70:73], a[128:131], a[32:35], v[70:73]
ds_read_b128 a[230:233], v255, offset:50752
v_mfma_f32_16x16x32_f16 v[70:73], a[132:135], a[36:39], v[70:73]
ds_read_b128 a[234:237], v255, offset:50816
v_mfma_f32_16x16x32_f16 v[66:69], a[136:139], a[32:35], v[66:69]
ds_read_b128 a[238:241], v255, offset:50880
v_mfma_f32_16x16x32_f16 v[66:69], a[140:143], a[36:39], v[66:69]
ds_read_b128 v[26:29], v255, offset:51200
v_mfma_f32_16x16x32_f16 v[62:65], a[104:107], a[24:27], v[62:65]
ds_read_b128 v[48:51], v255, offset:51264
v_mfma_f32_16x16x32_f16 v[62:65], a[108:111], a[28:31], v[62:65]
ds_read_b128 v[52:55], v255, offset:51328
v_mfma_f32_16x16x32_f16 v[58:61], a[112:115], a[24:27], v[58:61]
ds_read_b128 v[90:93], v255, offset:51392
v_mfma_f32_16x16x32_f16 v[58:61], a[116:119], a[28:31], v[58:61]
ds_read_b128 a[88:91], v46
v_mfma_f32_16x16x32_f16 a[170:173], a[128:131], a[24:27], a[170:173]
ds_read_b128 a[92:95], v46, offset:64
v_mfma_f32_16x16x32_f16 a[170:173], a[132:135], a[28:31], a[170:173]
ds_read_b128 a[80:83], v46, offset:128
v_mfma_f32_16x16x32_f16 a[174:177], a[136:139], a[24:27], a[174:177]
ds_read_b128 a[84:87], v46, offset:192
v_mfma_f32_16x16x32_f16 a[174:177], a[140:143], a[28:31], a[174:177]
ds_read_b128 a[72:75], v46, offset:512
v_mfma_f32_16x16x32_f16 v[98:101], a[104:107], a[16:19], v[98:101]
ds_read_b128 a[76:79], v46, offset:576
v_mfma_f32_16x16x32_f16 v[98:101], a[108:111], a[20:23], v[98:101]
ds_read_b128 a[64:67], v46, offset:640
v_mfma_f32_16x16x32_f16 v[94:97], a[112:115], a[16:19], v[94:97]
ds_read_b128 a[68:71], v46, offset:704
v_mfma_f32_16x16x32_f16 v[94:97], a[116:119], a[20:23], v[94:97]
v_mfma_f32_16x16x32_f16 v[242:245], a[128:131], a[16:19], v[242:245]
v_mfma_f32_16x16x32_f16 v[242:245], a[132:135], a[20:23], v[242:245]
s_add_u32 s16, s20, 0x80
s_addc_u32 s11, s21, 0
s_and_b32 s17, s11, 0xffff
v_cndmask_b32_e32 v1, v1, v21, vcc
s_mov_b32 m0, s35
v_mfma_f32_16x16x32_f16 a[124:127], a[136:139], a[16:19], a[124:127]
buffer_load_dwordx4 v1, s[16:19], 0, offen, lds
v_mfma_f32_16x16x32_f16 a[124:127], a[140:143], a[20:23], a[124:127]
v_mfma_f32_16x16x32_f16 a[120:123], a[104:107], a[8:11], a[120:123]
v_mfma_f32_16x16x32_f16 a[120:123], a[108:111], a[12:15], a[120:123]
v_cndmask_b32_e32 v22, v22, v21, vcc
s_mov_b32 m0, s36
v_mfma_f32_16x16x32_f16 a[250:253], a[112:115], a[8:11], a[250:253]
buffer_load_dwordx4 v22, s[16:19], 0, offen, lds
v_mfma_f32_16x16x32_f16 a[250:253], a[116:119], a[12:15], a[250:253]
v_mfma_f32_16x16x32_f16 a[246:249], a[128:131], a[8:11], a[246:249]
v_mfma_f32_16x16x32_f16 a[246:249], a[132:135], a[12:15], a[246:249]
v_cndmask_b32_e32 v23, v23, v21, vcc
s_mov_b32 m0, s37
v_mfma_f32_16x16x32_f16 a[242:245], a[136:139], a[8:11], a[242:245]
buffer_load_dwordx4 v23, s[16:19], 0, offen, lds
v_mfma_f32_16x16x32_f16 a[242:245], a[140:143], a[12:15], a[242:245]
v_mfma_f32_16x16x32_f16 a[206:209], a[104:107], a[0:3], a[206:209]
v_mfma_f32_16x16x32_f16 a[206:209], a[108:111], a[4:7], a[206:209]
v_cndmask_b32_e32 v44, v44, v21, vcc
s_mov_b32 m0, s38
v_mfma_f32_16x16x32_f16 a[222:225], a[112:115], a[0:3], a[222:225]
buffer_load_dwordx4 v44, s[16:19], 0, offen, lds
v_mfma_f32_16x16x32_f16 a[222:225], a[116:119], a[4:7], a[222:225]
v_mfma_f32_16x16x32_f16 a[218:221], a[128:131], a[0:3], a[218:221]
v_mfma_f32_16x16x32_f16 a[218:221], a[132:135], a[4:7], a[218:221]
v_mfma_f32_16x16x32_f16 v[250:253], a[136:139], a[0:3], v[250:253]
v_mfma_f32_16x16x32_f16 v[250:253], a[140:143], a[4:7], v[250:253]
s_waitcnt lgkmcnt(0)
v_mfma_f32_16x16x32_f16 a[96:99], a[88:91], a[144:147], a[96:99]
v_mfma_f32_16x16x32_f16 a[96:99], a[92:95], a[148:151], a[96:99]
v_mfma_f32_16x16x32_f16 v[30:33], a[80:83], a[144:147], v[30:33]
v_mfma_f32_16x16x32_f16 v[30:33], a[84:87], a[148:151], v[30:33]
s_waitcnt vmcnt(16), lgkmcnt(0)
s_barrier
ds_read_b128 a[104:107], v2
v_mfma_f32_16x16x32_f16 v[130:133], a[72:75], a[144:147], v[130:133]
ds_read_b128 a[108:111], v2, offset:64
v_mfma_f32_16x16x32_f16 v[130:133], a[76:79], a[148:151], v[130:133]
ds_read_b128 a[112:115], v2, offset:128
v_mfma_f32_16x16x32_f16 a[100:103], a[64:67], a[144:147], a[100:103]
ds_read_b128 a[116:119], v2, offset:192
v_mfma_f32_16x16x32_f16 a[100:103], a[68:71], a[148:151], a[100:103]
ds_read_b128 a[128:131], v2, offset:512
v_mfma_f32_16x16x32_f16 a[186:189], a[88:91], a[158:161], a[186:189]
ds_read_b128 a[132:135], v2, offset:576
v_mfma_f32_16x16x32_f16 a[186:189], a[92:95], a[178:181], a[186:189]
ds_read_b128 a[136:139], v2, offset:640
v_mfma_f32_16x16x32_f16 a[190:193], a[80:83], a[158:161], a[190:193]
ds_read_b128 a[140:143], v2, offset:704
v_mfma_f32_16x16x32_f16 a[190:193], a[84:87], a[178:181], a[190:193]
v_mfma_f32_16x16x32_f16 v[230:233], a[72:75], a[158:161], v[230:233]
v_mfma_f32_16x16x32_f16 v[230:233], a[76:79], a[178:181], v[230:233]
s_and_b32 s57, s3, 0xffff
s_mov_b32 s56, s2
s_mov_b32 m0, s39
v_mfma_f32_16x16x32_f16 v[226:229], a[64:67], a[158:161], v[226:229]
buffer_load_dwordx4 v20, s[56:59], 0, offen, lds
v_mfma_f32_16x16x32_f16 v[226:229], a[68:71], a[178:181], v[226:229]
v_mfma_f32_16x16x32_f16 v[222:225], a[88:91], a[182:185], v[222:225]
v_mfma_f32_16x16x32_f16 v[222:225], a[92:95], a[194:197], v[222:225]
s_mov_b32 m0, s40
s_nop 0
buffer_load_dwordx4 v37, s[56:59], 0, offen, lds
v_accvgpr_read_b32 v254, a157
v_mfma_f32_16x16x32_f16 v[218:221], a[80:83], a[182:185], v[218:221]
v_mfma_f32_16x16x32_f16 v[218:221], a[84:87], a[194:197], v[218:221]
v_mfma_f32_16x16x32_f16 v[214:217], a[72:75], a[182:185], v[214:217]
v_mfma_f32_16x16x32_f16 v[214:217], a[76:79], a[194:197], v[214:217]
s_mov_b32 m0, s41
v_mfma_f32_16x16x32_f16 v[210:213], a[64:67], a[182:185], v[210:213]
buffer_load_dwordx4 v42, s[56:59], 0, offen, lds
v_mfma_f32_16x16x32_f16 v[210:213], a[68:71], a[194:197], v[210:213]
v_mfma_f32_16x16x32_f16 v[206:209], a[88:91], a[198:201], v[206:209]
v_mfma_f32_16x16x32_f16 v[206:209], a[92:95], a[202:205], v[206:209]
s_mov_b32 m0, s42
v_mfma_f32_16x16x32_f16 v[202:205], a[80:83], a[198:201], v[202:205]
buffer_load_dwordx4 v40, s[56:59], 0, offen, lds
v_mfma_f32_16x16x32_f16 v[202:205], a[84:87], a[202:205], v[202:205]
v_mfma_f32_16x16x32_f16 v[198:201], a[72:75], a[198:201], v[198:201]
v_mfma_f32_16x16x32_f16 v[198:201], a[76:79], a[202:205], v[198:201]
s_mov_b32 m0, s43
v_mfma_f32_16x16x32_f16 v[194:197], a[64:67], a[198:201], v[194:197]
buffer_load_dwordx4 v45, s[56:59], 0, offen, lds
v_mfma_f32_16x16x32_f16 v[194:197], a[68:71], a[202:205], v[194:197]
v_mfma_f32_16x16x32_f16 v[190:193], a[88:91], a[226:229], v[190:193]
v_mfma_f32_16x16x32_f16 v[190:193], a[92:95], a[230:233], v[190:193]
s_mov_b32 m0, s44
v_mfma_f32_16x16x32_f16 v[186:189], a[80:83], a[226:229], v[186:189]
buffer_load_dwordx4 v0, s[56:59], 0, offen, lds
v_mfma_f32_16x16x32_f16 v[186:189], a[84:87], a[230:233], v[186:189]
v_mfma_f32_16x16x32_f16 v[182:185], a[72:75], a[226:229], v[182:185]
v_mfma_f32_16x16x32_f16 v[182:185], a[76:79], a[230:233], v[182:185]
s_mov_b32 m0, s45
v_mfma_f32_16x16x32_f16 v[178:181], a[64:67], a[226:229], v[178:181]
buffer_load_dwordx4 v36, s[56:59], 0, offen, lds
v_mfma_f32_16x16x32_f16 v[178:181], a[68:71], a[230:233], v[178:181]
v_mfma_f32_16x16x32_f16 v[174:177], a[88:91], a[234:237], v[174:177]
v_mfma_f32_16x16x32_f16 v[174:177], a[92:95], a[238:241], v[174:177]
s_mov_b32 m0, s46
v_mfma_f32_16x16x32_f16 v[170:173], a[80:83], a[234:237], v[170:173]
buffer_load_dwordx4 v41, s[56:59], 0, offen, lds
v_mfma_f32_16x16x32_f16 v[170:173], a[84:87], a[238:241], v[170:173]
v_mfma_f32_16x16x32_f16 v[166:169], a[72:75], a[234:237], v[166:169]
v_mfma_f32_16x16x32_f16 v[166:169], a[76:79], a[238:241], v[166:169]
s_mov_b32 m0, s47
v_mfma_f32_16x16x32_f16 v[162:165], a[64:67], a[234:237], v[162:165]
buffer_load_dwordx4 v38, s[16:19], 0, offen, lds
v_mfma_f32_16x16x32_f16 v[162:165], a[68:71], a[238:241], v[162:165]
v_mfma_f32_16x16x32_f16 v[158:161], a[88:91], v[26:29], v[158:161]
v_mfma_f32_16x16x32_f16 v[158:161], a[92:95], v[48:51], v[158:161]
s_mov_b32 m0, s48
v_mfma_f32_16x16x32_f16 v[154:157], a[80:83], v[26:29], v[154:157]
buffer_load_dwordx4 v43, s[16:19], 0, offen, lds
v_mfma_f32_16x16x32_f16 v[154:157], a[84:87], v[48:51], v[154:157]
v_mfma_f32_16x16x32_f16 v[150:153], a[72:75], v[26:29], v[150:153]
v_mfma_f32_16x16x32_f16 v[150:153], a[76:79], v[48:51], v[150:153]
s_mov_b32 m0, s49
v_mfma_f32_16x16x32_f16 v[146:149], a[64:67], v[26:29], v[146:149]
buffer_load_dwordx4 v34, s[16:19], 0, offen, lds
v_mfma_f32_16x16x32_f16 v[146:149], a[68:71], v[48:51], v[146:149]
v_mfma_f32_16x16x32_f16 v[142:145], a[88:91], v[52:55], v[142:145]
v_mfma_f32_16x16x32_f16 v[142:145], a[92:95], v[90:93], v[142:145]
s_mov_b32 m0, s50
v_mfma_f32_16x16x32_f16 v[138:141], a[80:83], v[52:55], v[138:141]
buffer_load_dwordx4 v39, s[16:19], 0, offen, lds
v_mfma_f32_16x16x32_f16 v[138:141], a[84:87], v[90:93], v[138:141]
v_mfma_f32_16x16x32_f16 v[134:137], a[72:75], v[52:55], v[134:137]
v_mfma_f32_16x16x32_f16 v[134:137], a[76:79], v[90:93], v[134:137]
v_mfma_f32_16x16x32_f16 v[6:9], a[64:67], v[52:55], v[6:9]
v_mfma_f32_16x16x32_f16 v[6:9], a[68:71], v[90:93], v[6:9]
s_waitcnt lgkmcnt(0)
v_mfma_f32_16x16x32_f16 v[126:129], a[104:107], a[144:147], v[126:129]
v_mfma_f32_16x16x32_f16 v[126:129], a[108:111], a[148:151], v[126:129]
v_mfma_f32_16x16x32_f16 v[122:125], a[112:115], a[144:147], v[122:125]
v_mfma_f32_16x16x32_f16 v[122:125], a[116:119], a[148:151], v[122:125]
v_mfma_f32_16x16x32_f16 v[118:121], a[128:131], a[144:147], v[118:121]
v_mfma_f32_16x16x32_f16 v[118:121], a[132:135], a[148:151], v[118:121]
v_mfma_f32_16x16x32_f16 v[114:117], a[136:139], a[144:147], v[114:117]
v_mfma_f32_16x16x32_f16 v[114:117], a[140:143], a[148:151], v[114:117]
v_mfma_f32_16x16x32_f16 v[110:113], a[104:107], a[158:161], v[110:113]
v_mfma_f32_16x16x32_f16 v[110:113], a[108:111], a[178:181], v[110:113]
v_mfma_f32_16x16x32_f16 a[162:165], a[112:115], a[158:161], a[162:165]
v_mfma_f32_16x16x32_f16 a[162:165], a[116:119], a[178:181], a[162:165]
v_mfma_f32_16x16x32_f16 a[166:169], a[128:131], a[158:161], a[166:169]
v_mfma_f32_16x16x32_f16 a[166:169], a[132:135], a[178:181], a[166:169]
v_mfma_f32_16x16x32_f16 a[214:217], a[136:139], a[158:161], a[214:217]
v_mfma_f32_16x16x32_f16 a[214:217], a[140:143], a[178:181], a[214:217]
v_mfma_f32_16x16x32_f16 a[210:213], a[104:107], a[182:185], a[210:213]
v_mfma_f32_16x16x32_f16 a[210:213], a[108:111], a[194:197], a[210:213]
v_mfma_f32_16x16x32_f16 v[246:249], a[112:115], a[182:185], v[246:249]
v_mfma_f32_16x16x32_f16 v[246:249], a[116:119], a[194:197], v[246:249]
s_waitcnt vmcnt(16), lgkmcnt(0)
s_barrier
ds_read_b128 a[56:59], v255
v_mfma_f32_16x16x32_f16 v[86:89], a[128:131], a[182:185], v[86:89]
ds_read_b128 a[60:63], v255, offset:64
v_mfma_f32_16x16x32_f16 v[86:89], a[132:135], a[194:197], v[86:89]
ds_read_b128 a[48:51], v255, offset:128
v_mfma_f32_16x16x32_f16 v[82:85], a[136:139], a[182:185], v[82:85]
ds_read_b128 a[52:55], v255, offset:192
v_mfma_f32_16x16x32_f16 v[82:85], a[140:143], a[194:197], v[82:85]
ds_read_b128 a[40:43], v255, offset:512
v_mfma_f32_16x16x32_f16 v[78:81], a[104:107], a[198:201], v[78:81]
ds_read_b128 a[44:47], v255, offset:576
v_mfma_f32_16x16x32_f16 v[78:81], a[108:111], a[202:205], v[78:81]
ds_read_b128 a[32:35], v255, offset:640
v_mfma_f32_16x16x32_f16 v[74:77], a[112:115], a[198:201], v[74:77]
ds_read_b128 a[36:39], v255, offset:704
v_mfma_f32_16x16x32_f16 v[74:77], a[116:119], a[202:205], v[74:77]
ds_read_b128 a[24:27], v255, offset:16896
v_mfma_f32_16x16x32_f16 v[70:73], a[128:131], a[198:201], v[70:73]
ds_read_b128 a[28:31], v255, offset:16960
v_mfma_f32_16x16x32_f16 v[70:73], a[132:135], a[202:205], v[70:73]
ds_read_b128 a[16:19], v255, offset:17024
v_mfma_f32_16x16x32_f16 v[66:69], a[136:139], a[198:201], v[66:69]
ds_read_b128 a[20:23], v255, offset:17088
v_mfma_f32_16x16x32_f16 v[66:69], a[140:143], a[202:205], v[66:69]
ds_read_b128 a[8:11], v255, offset:17408
v_mfma_f32_16x16x32_f16 v[62:65], a[104:107], a[226:229], v[62:65]
ds_read_b128 a[12:15], v255, offset:17472
v_mfma_f32_16x16x32_f16 v[62:65], a[108:111], a[230:233], v[62:65]
ds_read_b128 a[0:3], v255, offset:17536
v_mfma_f32_16x16x32_f16 v[58:61], a[112:115], a[226:229], v[58:61]
ds_read_b128 a[4:7], v255, offset:17600
v_mfma_f32_16x16x32_f16 v[58:61], a[116:119], a[230:233], v[58:61]
ds_read_b128 a[88:91], v3
v_mfma_f32_16x16x32_f16 a[170:173], a[128:131], a[226:229], a[170:173]
ds_read_b128 a[92:95], v3, offset:64
v_mfma_f32_16x16x32_f16 a[170:173], a[132:135], a[230:233], a[170:173]
ds_read_b128 a[80:83], v3, offset:128
v_mfma_f32_16x16x32_f16 a[174:177], a[136:139], a[226:229], a[174:177]
ds_read_b128 a[84:87], v3, offset:192
v_mfma_f32_16x16x32_f16 a[174:177], a[140:143], a[230:233], a[174:177]
ds_read_b128 a[72:75], v3, offset:512
v_mfma_f32_16x16x32_f16 v[98:101], a[104:107], a[234:237], v[98:101]
ds_read_b128 a[76:79], v3, offset:576
v_mfma_f32_16x16x32_f16 v[98:101], a[108:111], a[238:241], v[98:101]
ds_read_b128 a[64:67], v3, offset:640
v_mfma_f32_16x16x32_f16 v[94:97], a[112:115], a[234:237], v[94:97]
ds_read_b128 a[68:71], v3, offset:704
v_mfma_f32_16x16x32_f16 v[94:97], a[116:119], a[238:241], v[94:97]
v_mfma_f32_16x16x32_f16 v[242:245], a[128:131], a[234:237], v[242:245]
v_mfma_f32_16x16x32_f16 v[242:245], a[132:135], a[238:241], v[242:245]
s_add_u32 s20, s20, 0x100
s_addc_u32 s21, s21, 0
s_and_b32 s17, s21, 0xffff
s_mov_b32 s16, s20
s_mov_b32 m0, s22
v_mfma_f32_16x16x32_f16 a[124:127], a[136:139], a[234:237], a[124:127]
buffer_load_dwordx4 v1, s[16:19], 0, offen, lds
v_mfma_f32_16x16x32_f16 a[124:127], a[140:143], a[238:241], a[124:127]
v_mfma_f32_16x16x32_f16 a[120:123], a[104:107], v[26:29], a[120:123]
v_mfma_f32_16x16x32_f16 a[120:123], a[108:111], v[48:51], a[120:123]
s_mov_b32 m0, s23
v_mfma_f32_16x16x32_f16 a[250:253], a[112:115], v[26:29], a[250:253]
buffer_load_dwordx4 v22, s[16:19], 0, offen, lds
v_mfma_f32_16x16x32_f16 a[250:253], a[116:119], v[48:51], a[250:253]
v_mfma_f32_16x16x32_f16 a[246:249], a[128:131], v[26:29], a[246:249]
v_mfma_f32_16x16x32_f16 a[246:249], a[132:135], v[48:51], a[246:249]
s_mov_b32 m0, s51
v_mfma_f32_16x16x32_f16 a[242:245], a[136:139], v[26:29], a[242:245]
buffer_load_dwordx4 v23, s[16:19], 0, offen, lds
v_mfma_f32_16x16x32_f16 a[242:245], a[140:143], v[48:51], a[242:245]
v_mfma_f32_16x16x32_f16 a[206:209], a[104:107], v[52:55], a[206:209]
v_mfma_f32_16x16x32_f16 a[206:209], a[108:111], v[90:93], a[206:209]
s_mov_b32 m0, s52
v_mfma_f32_16x16x32_f16 a[222:225], a[112:115], v[52:55], a[222:225]
buffer_load_dwordx4 v44, s[16:19], 0, offen, lds
v_mfma_f32_16x16x32_f16 a[222:225], a[116:119], v[90:93], a[222:225]
v_mfma_f32_16x16x32_f16 a[218:221], a[128:131], v[52:55], a[218:221]
v_mfma_f32_16x16x32_f16 a[218:221], a[132:135], v[90:93], a[218:221]
v_mfma_f32_16x16x32_f16 v[250:253], a[136:139], v[52:55], v[250:253]
v_mfma_f32_16x16x32_f16 v[250:253], a[140:143], v[90:93], v[250:253]
s_add_i32 s24, s24, 2
s_add_u32 s2, s2, 0x100
s_addc_u32 s3, s3, 0
s_cmp_lt_i32 s24, s9
s_cbranch_scc1 .LBB0_2
; %bb.3:
v_accvgpr_mov_b32 a159, a103
v_accvgpr_mov_b32 a158, a102
v_accvgpr_mov_b32 a157, a101
v_accvgpr_mov_b32 a156, a100
v_accvgpr_mov_b32 a144, a214
v_accvgpr_mov_b32 a145, a215
v_accvgpr_mov_b32 a146, a216
v_accvgpr_mov_b32 a147, a217
v_accvgpr_mov_b32 a140, a210
v_accvgpr_mov_b32 a141, a211
v_accvgpr_mov_b32 a142, a212
v_accvgpr_mov_b32 a143, a213
v_accvgpr_read_b32 v36, a155
v_accvgpr_read_b32 v34, a152
v_accvgpr_read_b32 v35, a153
v_accvgpr_read_b32 v4, a154
v_accvgpr_read_b32 v50, a174
v_accvgpr_read_b32 v51, a175
v_accvgpr_read_b32 v52, a176
v_accvgpr_read_b32 v53, a177
v_accvgpr_read_b32 v22, a190
v_accvgpr_read_b32 v23, a191
v_accvgpr_read_b32 v24, a192
v_accvgpr_read_b32 v25, a193
v_accvgpr_read_b32 v26, a186
v_accvgpr_read_b32 v27, a187
v_accvgpr_read_b32 v28, a188
v_accvgpr_read_b32 v29, a189
v_accvgpr_read_b32 v102, a166
v_accvgpr_read_b32 v103, a167
v_accvgpr_read_b32 v104, a168
v_accvgpr_read_b32 v105, a169
v_accvgpr_read_b32 v42, a170
v_accvgpr_read_b32 v43, a171
v_accvgpr_read_b32 v44, a172
v_accvgpr_read_b32 v45, a173
v_accvgpr_read_b32 v106, a162
v_accvgpr_read_b32 v107, a163
v_accvgpr_read_b32 v108, a164
v_accvgpr_read_b32 v109, a165
s_branch .LBB0_5
.LBB0_4:
v_mov_b32_e32 v253, 0
v_mov_b32_e32 v252, v253
v_mov_b32_e32 v251, v253
v_mov_b32_e32 v250, v253
v_accvgpr_write_b32 a221, v253
v_accvgpr_write_b32 a220, v253
v_accvgpr_write_b32 a219, v253
v_accvgpr_write_b32 a218, v253
v_accvgpr_write_b32 a225, v253
v_accvgpr_write_b32 a224, v253
v_accvgpr_write_b32 a223, v253
v_accvgpr_write_b32 a222, v253
v_accvgpr_write_b32 a209, v253
v_accvgpr_write_b32 a208, v253
v_accvgpr_write_b32 a207, v253
v_accvgpr_write_b32 a206, v253
v_accvgpr_write_b32 a245, v253
v_accvgpr_write_b32 a244, v253
v_accvgpr_write_b32 a243, v253
v_accvgpr_write_b32 a242, v253
v_accvgpr_write_b32 a249, v253
v_accvgpr_write_b32 a248, v253
v_accvgpr_write_b32 a247, v253
v_accvgpr_write_b32 a246, v253
v_accvgpr_write_b32 a253, v253
v_accvgpr_write_b32 a252, v253
v_accvgpr_write_b32 a251, v253
v_accvgpr_write_b32 a250, v253
v_accvgpr_write_b32 a123, v253
v_accvgpr_write_b32 a122, v253
v_accvgpr_write_b32 a121, v253
v_accvgpr_write_b32 a120, v253
v_accvgpr_write_b32 a127, v253
v_accvgpr_write_b32 a126, v253
v_accvgpr_write_b32 a125, v253
v_accvgpr_write_b32 a124, v253
v_mov_b32_e32 v245, v253
v_mov_b32_e32 v244, v253
v_mov_b32_e32 v243, v253
v_mov_b32_e32 v242, v253
v_mov_b32_e32 v97, v253
v_mov_b32_e32 v96, v253
v_mov_b32_e32 v95, v253
v_mov_b32_e32 v94, v253
v_mov_b32_e32 v101, v253
v_mov_b32_e32 v100, v253
v_mov_b32_e32 v99, v253
v_mov_b32_e32 v98, v253
v_mov_b32_e32 v53, v253
v_mov_b32_e32 v52, v253
v_mov_b32_e32 v51, v253
v_mov_b32_e32 v50, v253
v_mov_b32_e32 v45, v253
v_mov_b32_e32 v44, v253
v_mov_b32_e32 v43, v253
v_mov_b32_e32 v42, v253
v_mov_b32_e32 v61, v253
v_mov_b32_e32 v60, v253
v_mov_b32_e32 v59, v253
v_mov_b32_e32 v58, v253
v_mov_b32_e32 v65, v253
v_mov_b32_e32 v64, v253
v_mov_b32_e32 v63, v253
v_mov_b32_e32 v62, v253
v_mov_b32_e32 v69, v253
v_mov_b32_e32 v68, v253
v_mov_b32_e32 v67, v253
v_mov_b32_e32 v66, v253
v_mov_b32_e32 v73, v253
v_mov_b32_e32 v72, v253
v_mov_b32_e32 v71, v253
v_mov_b32_e32 v70, v253
v_mov_b32_e32 v77, v253
v_mov_b32_e32 v76, v253
v_mov_b32_e32 v75, v253
v_mov_b32_e32 v74, v253
v_mov_b32_e32 v81, v253
v_mov_b32_e32 v80, v253
v_mov_b32_e32 v79, v253
v_mov_b32_e32 v78, v253
v_mov_b32_e32 v85, v253
v_mov_b32_e32 v84, v253
v_mov_b32_e32 v83, v253
v_mov_b32_e32 v82, v253
v_mov_b32_e32 v89, v253
v_mov_b32_e32 v88, v253
v_mov_b32_e32 v87, v253
v_mov_b32_e32 v86, v253
v_mov_b32_e32 v249, v253
v_mov_b32_e32 v248, v253
v_mov_b32_e32 v247, v253
v_mov_b32_e32 v246, v253
v_mov_b32_e32 v3, v253
v_mov_b32_e32 v2, v253
v_mov_b32_e32 v1, v253
v_mov_b32_e32 v0, v253
v_accvgpr_write_b32 a143, v3
v_accvgpr_write_b32 a142, v2
v_accvgpr_write_b32 a141, v1
v_accvgpr_write_b32 a140, v0
v_accvgpr_write_b32 a147, v3
v_accvgpr_write_b32 a146, v2
v_accvgpr_write_b32 a145, v1
v_accvgpr_write_b32 a144, v0
v_mov_b32_e32 v105, v253
v_mov_b32_e32 v104, v253
v_mov_b32_e32 v103, v253
v_mov_b32_e32 v102, v253
v_mov_b32_e32 v109, v253
v_mov_b32_e32 v108, v253
v_mov_b32_e32 v107, v253
v_mov_b32_e32 v106, v253
v_mov_b32_e32 v113, v253
v_mov_b32_e32 v112, v253
v_mov_b32_e32 v111, v253
v_mov_b32_e32 v110, v253
v_mov_b32_e32 v117, v253
v_mov_b32_e32 v116, v253
v_mov_b32_e32 v115, v253
v_mov_b32_e32 v114, v253
v_mov_b32_e32 v121, v253
v_mov_b32_e32 v120, v253
v_mov_b32_e32 v119, v253
v_mov_b32_e32 v118, v253
v_mov_b32_e32 v125, v253
v_mov_b32_e32 v124, v253
v_mov_b32_e32 v123, v253
v_mov_b32_e32 v122, v253
v_mov_b32_e32 v129, v253
v_mov_b32_e32 v128, v253
v_mov_b32_e32 v127, v253
v_mov_b32_e32 v126, v253
v_mov_b32_e32 v9, v253
v_mov_b32_e32 v8, v253
v_mov_b32_e32 v7, v253
v_mov_b32_e32 v6, v253
v_mov_b32_e32 v137, v253
v_mov_b32_e32 v136, v253
v_mov_b32_e32 v135, v253
v_mov_b32_e32 v134, v253
v_mov_b32_e32 v141, v253
v_mov_b32_e32 v140, v253
v_mov_b32_e32 v139, v253
v_mov_b32_e32 v138, v253
v_mov_b32_e32 v145, v253
v_mov_b32_e32 v144, v253
v_mov_b32_e32 v143, v253
v_mov_b32_e32 v142, v253
v_mov_b32_e32 v149, v253
v_mov_b32_e32 v148, v253
v_mov_b32_e32 v147, v253
v_mov_b32_e32 v146, v253
v_mov_b32_e32 v153, v253
v_mov_b32_e32 v152, v253
v_mov_b32_e32 v151, v253
v_mov_b32_e32 v150, v253
v_mov_b32_e32 v157, v253
v_mov_b32_e32 v156, v253
v_mov_b32_e32 v155, v253
v_mov_b32_e32 v154, v253
v_mov_b32_e32 v161, v253
v_mov_b32_e32 v160, v253
v_mov_b32_e32 v159, v253
v_mov_b32_e32 v158, v253
v_mov_b32_e32 v165, v253
v_mov_b32_e32 v164, v253
v_mov_b32_e32 v163, v253
v_mov_b32_e32 v162, v253
v_mov_b32_e32 v169, v253
v_mov_b32_e32 v168, v253
v_mov_b32_e32 v167, v253
v_mov_b32_e32 v166, v253
v_mov_b32_e32 v173, v253
v_mov_b32_e32 v172, v253
v_mov_b32_e32 v171, v253
v_mov_b32_e32 v170, v253
v_mov_b32_e32 v177, v253
v_mov_b32_e32 v176, v253
v_mov_b32_e32 v175, v253
v_mov_b32_e32 v174, v253
v_mov_b32_e32 v181, v253
v_mov_b32_e32 v180, v253
v_mov_b32_e32 v179, v253
v_mov_b32_e32 v178, v253
v_mov_b32_e32 v185, v253
v_mov_b32_e32 v184, v253
v_mov_b32_e32 v183, v253
v_mov_b32_e32 v182, v253
v_mov_b32_e32 v189, v253
v_mov_b32_e32 v188, v253
v_mov_b32_e32 v187, v253
v_mov_b32_e32 v186, v253
v_mov_b32_e32 v193, v253
v_mov_b32_e32 v192, v253
v_mov_b32_e32 v191, v253
v_mov_b32_e32 v190, v253
v_mov_b32_e32 v197, v253
v_mov_b32_e32 v196, v253
v_mov_b32_e32 v195, v253
v_mov_b32_e32 v194, v253
v_mov_b32_e32 v201, v253
v_mov_b32_e32 v200, v253
v_mov_b32_e32 v199, v253
v_mov_b32_e32 v198, v253
v_mov_b32_e32 v205, v253
v_mov_b32_e32 v204, v253
v_mov_b32_e32 v203, v253
v_mov_b32_e32 v202, v253
v_mov_b32_e32 v209, v253
v_mov_b32_e32 v208, v253
v_mov_b32_e32 v207, v253
v_mov_b32_e32 v206, v253
v_mov_b32_e32 v213, v253
v_mov_b32_e32 v212, v253
v_mov_b32_e32 v211, v253
v_mov_b32_e32 v210, v253
v_mov_b32_e32 v217, v253
v_mov_b32_e32 v216, v253
v_mov_b32_e32 v215, v253
v_mov_b32_e32 v214, v253
v_mov_b32_e32 v221, v253
v_mov_b32_e32 v220, v253
v_mov_b32_e32 v219, v253
v_mov_b32_e32 v218, v253
v_mov_b32_e32 v225, v253
v_mov_b32_e32 v224, v253
v_mov_b32_e32 v223, v253
v_mov_b32_e32 v222, v253
v_mov_b32_e32 v229, v253
v_mov_b32_e32 v228, v253
v_mov_b32_e32 v227, v253
v_mov_b32_e32 v226, v253
v_mov_b32_e32 v233, v253
v_mov_b32_e32 v232, v253
v_mov_b32_e32 v231, v253
v_mov_b32_e32 v230, v253
v_mov_b32_e32 v25, v253
v_mov_b32_e32 v24, v253
v_mov_b32_e32 v23, v253
v_mov_b32_e32 v22, v253
v_mov_b32_e32 v29, v253
v_mov_b32_e32 v28, v253
v_mov_b32_e32 v27, v253
v_mov_b32_e32 v26, v253
v_accvgpr_write_b32 a159, v3
v_accvgpr_write_b32 a158, v2
v_accvgpr_write_b32 a157, v1
v_accvgpr_write_b32 a156, v0
v_mov_b32_e32 v133, v253
v_mov_b32_e32 v132, v253
v_mov_b32_e32 v131, v253
v_mov_b32_e32 v130, v253
v_mov_b32_e32 v33, v253
v_mov_b32_e32 v32, v253
v_mov_b32_e32 v31, v253
v_mov_b32_e32 v30, v253
v_accvgpr_write_b32 a99, v253
v_accvgpr_write_b32 a98, v253
v_accvgpr_write_b32 a97, v253
v_accvgpr_write_b32 a96, v253
.LBB0_5:
v_accvgpr_mov_b32 a155, a99
v_accvgpr_mov_b32 a154, a98
v_accvgpr_mov_b32 a153, a97
v_accvgpr_mov_b32 a152, a96
v_accvgpr_write_b32 a151, v9
v_accvgpr_write_b32 a150, v8
v_accvgpr_write_b32 a149, v7
v_accvgpr_write_b32 a148, v6
v_accvgpr_write_b32 a136, v246
v_accvgpr_write_b32 a137, v247
v_accvgpr_write_b32 a138, v248
v_accvgpr_write_b32 a139, v249
v_mov_b64_e32 v[38:39], v[98:99]
v_mov_b64_e32 v[40:41], v[100:101]
v_accvgpr_write_b32 a135, v97
v_accvgpr_write_b32 a134, v96
v_accvgpr_write_b32 a133, v95
v_accvgpr_write_b32 a132, v94
v_accvgpr_write_b32 a128, v242
v_accvgpr_write_b32 a129, v243
v_accvgpr_write_b32 a130, v244
v_accvgpr_write_b32 a131, v245
s_lshl_b32 s18, s1, 6
v_and_b32_e32 v10, 63, v34
v_or_b32_e32 v0, s18, v10
s_waitcnt vmcnt(0), lgkmcnt(0)
s_barrier
v_lshrrev_b32_e32 v0, 4, v0
v_or_b32_e32 v1, 16, v0
v_or_b32_e32 v2, 32, v0
v_or_b32_e32 v3, 48, v0
v_lshlrev_b32_e32 v11, 3, v4
v_mul_lo_u32 v12, v0, s13
v_mul_lo_u32 v13, v1, s13
v_mul_lo_u32 v14, v2, s13
v_mul_lo_u32 v15, v3, s13
s_mul_i32 s2, s8, s13
s_ashr_i32 s3, s2, 31
s_lshl_b64 s[2:3], s[2:3], 1
s_add_u32 s2, s6, s2
s_addc_u32 s3, s7, s3
s_ashr_i32 s1, s0, 31
s_lshl_b64 s[0:1], s[0:1], 1
s_add_u32 s4, s2, s0
s_addc_u32 s19, s3, s1
s_lshl_b32 s0, s13, 6
s_ashr_i32 s1, s0, 31
s_lshl_b64 s[0:1], s[0:1], 1
s_add_u32 s28, s4, s0
s_addc_u32 s17, s19, s1
s_add_u32 s24, s28, s0
s_addc_u32 s13, s17, s1
s_add_u32 s20, s24, s0
s_addc_u32 s11, s13, s1
s_add_u32 s16, s4, 0x100
s_addc_u32 s10, s19, 0
s_add_u32 s12, s28, 0x100
s_addc_u32 s3, s17, 0
s_add_u32 s8, s24, 0x100
s_addc_u32 s2, s13, 0
s_add_u32 s0, s20, 0x100
s_addc_u32 s1, s11, 0
s_lshr_b32 s6, s9, 31
s_add_i32 s6, s9, s6
s_and_b32 s6, s6, -2
s_sub_i32 s9, s9, s6
v_accvgpr_read_b32 v0, a152
v_accvgpr_read_b32 v1, a153
v_accvgpr_read_b32 v2, a154
v_accvgpr_read_b32 v3, a155
s_nop 1
v_mfma_f32_16x16x32_f16 v[0:3], a[88:91], a[56:59], v[0:3]
v_mfma_f32_16x16x32_f16 v[0:3], a[92:95], a[60:63], v[0:3]
v_mfma_f32_16x16x32_f16 v[4:7], a[80:83], a[56:59], v[30:33]
v_mfma_f32_16x16x32_f16 v[4:7], a[84:87], a[60:63], v[4:7]
v_mfma_f32_16x16x32_f16 v[130:133], a[72:75], a[56:59], v[130:133]
v_mfma_f32_16x16x32_f16 v[130:133], a[76:79], a[60:63], v[130:133]
v_accvgpr_read_b32 v30, a156
v_accvgpr_read_b32 v31, a157
v_accvgpr_read_b32 v32, a158
v_accvgpr_read_b32 v33, a159
s_nop 1
v_mfma_f32_16x16x32_f16 v[242:245], a[64:67], a[56:59], v[30:33]
v_mfma_f32_16x16x32_f16 v[242:245], a[68:71], a[60:63], v[242:245]
v_mfma_f32_16x16x32_f16 v[238:241], a[88:91], a[48:51], v[26:29]
v_mfma_f32_16x16x32_f16 v[238:241], a[92:95], a[52:55], v[238:241]
v_mfma_f32_16x16x32_f16 v[234:237], a[80:83], a[48:51], v[22:25]
v_mfma_f32_16x16x32_f16 v[234:237], a[84:87], a[52:55], v[234:237]
v_mfma_f32_16x16x32_f16 v[230:233], a[72:75], a[48:51], v[230:233]
v_mfma_f32_16x16x32_f16 v[230:233], a[76:79], a[52:55], v[230:233]
v_mfma_f32_16x16x32_f16 v[226:229], a[64:67], a[48:51], v[226:229]
v_mfma_f32_16x16x32_f16 v[226:229], a[68:71], a[52:55], v[226:229]
v_cvt_pk_f16_f32 v246, v0, v1
v_cvt_pk_f16_f32 v247, v2, v3
v_cvt_pk_f16_f32 v2, v4, v5
v_cvt_pk_f16_f32 v3, v6, v7
v_cvt_pk_f16_f32 v6, v130, v131
v_cvt_pk_f16_f32 v7, v132, v133
v_cvt_pk_f16_f32 v130, v242, v243
v_cvt_pk_f16_f32 v131, v244, v245
v_cvt_pk_f16_f32 v248, v238, v239
v_cvt_pk_f16_f32 v249, v240, v241
v_cvt_pk_f16_f32 v4, v234, v235
v_cvt_pk_f16_f32 v5, v236, v237
v_cvt_pk_f16_f32 v8, v230, v231
v_cvt_pk_f16_f32 v9, v232, v233
v_cvt_pk_f16_f32 v132, v226, v227
v_cvt_pk_f16_f32 v133, v228, v229
v_lshlrev_b32_e32 v0, 8, v34
v_and_b32_e32 v16, 0x70, v35
v_and_b32_e32 v17, 1, v34
v_lshlrev_b32_e32 v1, 12, v17
v_and_b32_e32 v18, 16, v34
v_lshlrev_b32_e32 v19, 4, v18
s_lshr_b32 s5, s5, 1
s_and_b64 s[6:7], s[14:15], exec
s_cselect_b32 s6, 0, 0x80
s_movk_i32 s7, 0x2e00
v_and_or_b32 v0, v0, s7, v1
v_or3_b32 v19, s6, v19, v0
v_mov_b32_e32 v0, 0x70
v_bitop3_b32 v20, s5, v35, v0, bitop3:0x78
v_or_b32_e32 v21, v19, v20
v_add_u32_e32 v0, 0, v21
ds_write_b128 v0, v[246:249]
v_xad_u32 v1, v21, 16, 0
ds_write_b128 v1, v[2:5]
v_xad_u32 v2, v21, 64, 0
ds_write_b128 v2, v[6:9]
s_movk_i32 s5, 0x50
v_bitop3_b32 v3, v19, s5, v20, bitop3:0x36
v_add_u32_e32 v3, 0, v3
ds_write_b128 v3, v[130:133]
s_waitcnt lgkmcnt(0)
s_barrier
v_mov_b32_e32 v4, 0xe0
v_bitop3_b32 v4, s18, v4, v10, bitop3:0xc8
v_lshlrev_b32_e32 v5, 4, v4
v_lshrrev_b32_e32 v4, 1, v4
v_lshlrev_b32_e32 v6, 8, v18
v_bitop3_b32 v4, v5, v4, v16, bitop3:0x36
v_lshl_add_u32 v5, v17, 13, 0
v_add3_u32 v4, v5, v6, v4
ds_read_b128 v[16:19], v4
ds_read_b128 v[22:25], v4, offset:256
ds_read_b128 v[26:29], v4, offset:128
ds_read_b128 v[32:35], v4, offset:384
s_and_b32 s5, s19, 0xffff
s_mov_b32 s7, 0x27000
s_mov_b32 s6, 0x7ffffffe
s_waitcnt lgkmcnt(3)
v_mov_b32_e32 v20, v16
v_mov_b32_e32 v21, v17
v_add_lshl_u32 v5, v12, v11, 1
s_waitcnt lgkmcnt(2)
buffer_store_dwordx4 v[20:23], v5, s[4:7], 0, offen
s_nop 1
v_mov_b32_e32 v20, v24
v_mov_b32_e32 v21, v25
v_add_lshl_u32 v6, v13, v11, 1
buffer_store_dwordx4 v[18:21], v6, s[4:7], 0, offen
s_waitcnt lgkmcnt(1)
v_mov_b32_e32 v30, v26
v_mov_b32_e32 v31, v27
v_add_lshl_u32 v7, v14, v11, 1
s_waitcnt lgkmcnt(0)
buffer_store_dwordx4 v[30:33], v7, s[4:7], 0, offen
s_nop 1
v_mov_b32_e32 v30, v34
v_mov_b32_e32 v31, v35
v_add_lshl_u32 v8, v15, v11, 1
buffer_store_dwordx4 v[28:31], v8, s[4:7], 0, offen
v_mfma_f32_16x16x32_f16 v[130:133], a[88:91], a[40:43], v[222:225]
v_mfma_f32_16x16x32_f16 v[130:133], a[92:95], a[44:47], v[130:133]
v_mfma_f32_16x16x32_f16 v[218:221], a[80:83], a[40:43], v[218:221]
v_mfma_f32_16x16x32_f16 v[218:221], a[84:87], a[44:47], v[218:221]
v_mfma_f32_16x16x32_f16 v[214:217], a[72:75], a[40:43], v[214:217]
v_mfma_f32_16x16x32_f16 v[214:217], a[76:79], a[44:47], v[214:217]
v_mfma_f32_16x16x32_f16 v[210:213], a[64:67], a[40:43], v[210:213]
v_mfma_f32_16x16x32_f16 v[210:213], a[68:71], a[44:47], v[210:213]
v_mfma_f32_16x16x32_f16 v[206:209], a[88:91], a[32:35], v[206:209]
v_mfma_f32_16x16x32_f16 v[206:209], a[92:95], a[36:39], v[206:209]
v_mfma_f32_16x16x32_f16 v[202:205], a[80:83], a[32:35], v[202:205]
v_mfma_f32_16x16x32_f16 v[202:205], a[84:87], a[36:39], v[202:205]
v_mfma_f32_16x16x32_f16 v[198:201], a[72:75], a[32:35], v[198:201]
v_mfma_f32_16x16x32_f16 v[198:201], a[76:79], a[36:39], v[198:201]
v_mfma_f32_16x16x32_f16 v[194:197], a[64:67], a[32:35], v[194:197]
v_mfma_f32_16x16x32_f16 v[194:197], a[68:71], a[36:39], v[194:197]
v_cvt_pk_f16_f32 v130, v130, v131
v_cvt_pk_f16_f32 v131, v132, v133
v_cvt_pk_f16_f32 v218, v218, v219
v_cvt_pk_f16_f32 v219, v220, v221
v_cvt_pk_f16_f32 v214, v214, v215
v_cvt_pk_f16_f32 v215, v216, v217
v_cvt_pk_f16_f32 v210, v210, v211
v_cvt_pk_f16_f32 v211, v212, v213
v_cvt_pk_f16_f32 v132, v206, v207
v_cvt_pk_f16_f32 v133, v208, v209
v_cvt_pk_f16_f32 v220, v202, v203
v_cvt_pk_f16_f32 v221, v204, v205
v_cvt_pk_f16_f32 v216, v198, v199
v_cvt_pk_f16_f32 v217, v200, v201
v_cvt_pk_f16_f32 v212, v194, v195
v_cvt_pk_f16_f32 v213, v196, v197
s_waitcnt lgkmcnt(0)
s_barrier
ds_write_b128 v0, v[130:133]
ds_write_b128 v1, v[218:221]
ds_write_b128 v2, v[214:217]
ds_write_b128 v3, v[210:213]
s_waitcnt lgkmcnt(0)
s_barrier
ds_read_b128 v[10:13], v4
ds_read_b128 v[16:19], v4, offset:256
ds_read_b128 v[20:23], v4, offset:128
ds_read_b128 v[26:29], v4, offset:384
s_and_b32 s29, s17, 0xffff
s_mov_b32 s30, s6
s_mov_b32 s31, s7
s_waitcnt lgkmcnt(3)
v_mov_b32_e32 v14, v10
v_mov_b32_e32 v15, v11
s_waitcnt lgkmcnt(2)
buffer_store_dwordx4 v[14:17], v5, s[28:31], 0, offen
s_nop 1
v_mov_b32_e32 v14, v18
v_mov_b32_e32 v15, v19
buffer_store_dwordx4 v[12:15], v6, s[28:31], 0, offen
s_waitcnt lgkmcnt(1)
v_mov_b32_e32 v24, v20
v_mov_b32_e32 v25, v21
s_waitcnt lgkmcnt(0)
buffer_store_dwordx4 v[24:27], v7, s[28:31], 0, offen
s_nop 1
v_mov_b32_e32 v24, v28
v_mov_b32_e32 v25, v29
buffer_store_dwordx4 v[22:25], v8, s[28:31], 0, offen
v_mfma_f32_16x16x32_f16 v[130:133], a[88:91], a[24:27], v[190:193]
v_mfma_f32_16x16x32_f16 v[130:133], a[92:95], a[28:31], v[130:133]
v_mfma_f32_16x16x32_f16 v[186:189], a[80:83], a[24:27], v[186:189]
v_mfma_f32_16x16x32_f16 v[186:189], a[84:87], a[28:31], v[186:189]
v_mfma_f32_16x16x32_f16 v[182:185], a[72:75], a[24:27], v[182:185]
v_mfma_f32_16x16x32_f16 v[182:185], a[76:79], a[28:31], v[182:185]
v_mfma_f32_16x16x32_f16 v[178:181], a[64:67], a[24:27], v[178:181]
v_mfma_f32_16x16x32_f16 v[178:181], a[68:71], a[28:31], v[178:181]
v_mfma_f32_16x16x32_f16 v[174:177], a[88:91], a[16:19], v[174:177]
v_mfma_f32_16x16x32_f16 v[174:177], a[92:95], a[20:23], v[174:177]
v_mfma_f32_16x16x32_f16 v[170:173], a[80:83], a[16:19], v[170:173]
v_mfma_f32_16x16x32_f16 v[170:173], a[84:87], a[20:23], v[170:173]
v_mfma_f32_16x16x32_f16 v[166:169], a[72:75], a[16:19], v[166:169]
v_mfma_f32_16x16x32_f16 v[166:169], a[76:79], a[20:23], v[166:169]
v_mfma_f32_16x16x32_f16 v[162:165], a[64:67], a[16:19], v[162:165]
v_mfma_f32_16x16x32_f16 v[162:165], a[68:71], a[20:23], v[162:165]
v_cvt_pk_f16_f32 v130, v130, v131
v_cvt_pk_f16_f32 v131, v132, v133
v_cvt_pk_f16_f32 v186, v186, v187
v_cvt_pk_f16_f32 v187, v188, v189
v_cvt_pk_f16_f32 v182, v182, v183
v_cvt_pk_f16_f32 v183, v184, v185
v_cvt_pk_f16_f32 v178, v178, v179
v_cvt_pk_f16_f32 v179, v180, v181
v_cvt_pk_f16_f32 v132, v174, v175
v_cvt_pk_f16_f32 v133, v176, v177
v_cvt_pk_f16_f32 v188, v170, v171
v_cvt_pk_f16_f32 v189, v172, v173
v_cvt_pk_f16_f32 v184, v166, v167
v_cvt_pk_f16_f32 v185, v168, v169
v_cvt_pk_f16_f32 v180, v162, v163
v_cvt_pk_f16_f32 v181, v164, v165
s_waitcnt lgkmcnt(0)
s_barrier
ds_write_b128 v0, v[130:133]
ds_write_b128 v1, v[186:189]
ds_write_b128 v2, v[182:185]
ds_write_b128 v3, v[178:181]
s_waitcnt lgkmcnt(0)
s_barrier
ds_read_b128 v[10:13], v4
ds_read_b128 v[16:19], v4, offset:256
ds_read_b128 v[20:23], v4, offset:128
ds_read_b128 v[26:29], v4, offset:384
s_and_b32 s25, s13, 0xffff
s_mov_b32 s26, s6
s_mov_b32 s27, s7
s_waitcnt lgkmcnt(3)
v_mov_b32_e32 v14, v10
v_mov_b32_e32 v15, v11
s_waitcnt lgkmcnt(2)
buffer_store_dwordx4 v[14:17], v5, s[24:27], 0, offen
s_nop 1
v_mov_b32_e32 v14, v18
v_mov_b32_e32 v15, v19
buffer_store_dwordx4 v[12:15], v6, s[24:27], 0, offen
s_waitcnt lgkmcnt(1)
v_mov_b32_e32 v24, v20
v_mov_b32_e32 v25, v21
s_waitcnt lgkmcnt(0)
buffer_store_dwordx4 v[24:27], v7, s[24:27], 0, offen
s_nop 1
v_mov_b32_e32 v24, v28
v_mov_b32_e32 v25, v29
buffer_store_dwordx4 v[22:25], v8, s[24:27], 0, offen
v_mfma_f32_16x16x32_f16 v[130:133], a[88:91], a[8:11], v[158:161]
v_mfma_f32_16x16x32_f16 v[130:133], a[92:95], a[12:15], v[130:133]
v_mfma_f32_16x16x32_f16 v[154:157], a[80:83], a[8:11], v[154:157]
v_mfma_f32_16x16x32_f16 v[154:157], a[84:87], a[12:15], v[154:157]
v_mfma_f32_16x16x32_f16 v[150:153], a[72:75], a[8:11], v[150:153]
v_mfma_f32_16x16x32_f16 v[150:153], a[76:79], a[12:15], v[150:153]
v_mfma_f32_16x16x32_f16 v[146:149], a[64:67], a[8:11], v[146:149]
v_mfma_f32_16x16x32_f16 v[146:149], a[68:71], a[12:15], v[146:149]
v_mfma_f32_16x16x32_f16 v[142:145], a[88:91], a[0:3], v[142:145]
v_mfma_f32_16x16x32_f16 v[142:145], a[92:95], a[4:7], v[142:145]
v_mfma_f32_16x16x32_f16 v[138:141], a[80:83], a[0:3], v[138:141]
v_mfma_f32_16x16x32_f16 v[138:141], a[84:87], a[4:7], v[138:141]
v_mfma_f32_16x16x32_f16 v[134:137], a[72:75], a[0:3], v[134:137]
v_mfma_f32_16x16x32_f16 v[134:137], a[76:79], a[4:7], v[134:137]
v_accvgpr_read_b32 v10, a148
v_accvgpr_read_b32 v11, a149
v_accvgpr_read_b32 v12, a150
v_accvgpr_read_b32 v13, a151
s_nop 1
v_mfma_f32_16x16x32_f16 v[158:161], a[64:67], a[0:3], v[10:13]
v_mfma_f32_16x16x32_f16 v[158:161], a[68:71], a[4:7], v[158:161]
v_cvt_pk_f16_f32 v130, v130, v131
v_cvt_pk_f16_f32 v131, v132, v133
v_cvt_pk_f16_f32 v154, v154, v155
v_cvt_pk_f16_f32 v155, v156, v157
v_cvt_pk_f16_f32 v150, v150, v151
v_cvt_pk_f16_f32 v151, v152, v153
v_cvt_pk_f16_f32 v146, v146, v147
v_cvt_pk_f16_f32 v147, v148, v149
v_cvt_pk_f16_f32 v132, v142, v143
v_cvt_pk_f16_f32 v133, v144, v145
v_cvt_pk_f16_f32 v156, v138, v139
v_cvt_pk_f16_f32 v157, v140, v141
v_cvt_pk_f16_f32 v152, v134, v135
v_cvt_pk_f16_f32 v153, v136, v137
v_cvt_pk_f16_f32 v148, v158, v159
v_cvt_pk_f16_f32 v149, v160, v161
s_waitcnt lgkmcnt(0)
s_barrier
ds_write_b128 v0, v[130:133]
ds_write_b128 v1, v[154:157]
ds_write_b128 v2, v[150:153]
ds_write_b128 v3, v[146:149]
s_waitcnt lgkmcnt(0)
s_barrier
ds_read_b128 v[10:13], v4
ds_read_b128 v[16:19], v4, offset:256
ds_read_b128 v[20:23], v4, offset:128
ds_read_b128 v[26:29], v4, offset:384
s_and_b32 s21, s11, 0xffff
s_mov_b32 s22, s6
s_mov_b32 s23, s7
s_waitcnt lgkmcnt(3)
v_mov_b32_e32 v14, v10
v_mov_b32_e32 v15, v11
s_waitcnt lgkmcnt(2)
buffer_store_dwordx4 v[14:17], v5, s[20:23], 0, offen
s_nop 1
v_mov_b32_e32 v14, v18
v_mov_b32_e32 v15, v19
buffer_store_dwordx4 v[12:15], v6, s[20:23], 0, offen
s_waitcnt lgkmcnt(1)
v_mov_b32_e32 v24, v20
v_mov_b32_e32 v25, v21
s_waitcnt lgkmcnt(0)
buffer_store_dwordx4 v[24:27], v7, s[20:23], 0, offen
s_nop 1
v_mov_b32_e32 v24, v28
v_mov_b32_e32 v25, v29
buffer_store_dwordx4 v[22:25], v8, s[20:23], 0, offen
s_lshl_b32 s4, s9, 14
s_add_i32 s4, s4, 0
s_lshl_b32 s5, s9, 9
s_and_b32 s5, s5, 0xffffe00
s_add_i32 s4, s4, s5
v_add_u32_e32 v9, s4, v36
v_add_u32_e32 v9, 0x18bc0, v9
ds_read_b128 v[130:133], v9
ds_read_b128 v[134:137], v9, offset:64
ds_read_b128 v[138:141], v9, offset:128
ds_read_b128 v[142:145], v9, offset:192
ds_read_b128 v[146:149], v9, offset:512
ds_read_b128 v[150:153], v9, offset:576
ds_read_b128 v[154:157], v9, offset:640
ds_read_b128 v[158:161], v9, offset:704
s_waitcnt lgkmcnt(7)
v_mfma_f32_16x16x32_f16 v[126:129], v[130:133], a[56:59], v[126:129]
s_waitcnt lgkmcnt(6)
v_mfma_f32_16x16x32_f16 v[126:129], v[134:137], a[60:63], v[126:129]
s_waitcnt lgkmcnt(5)
v_mfma_f32_16x16x32_f16 v[122:125], v[138:141], a[56:59], v[122:125]
s_waitcnt lgkmcnt(4)
v_mfma_f32_16x16x32_f16 v[122:125], v[142:145], a[60:63], v[122:125]
s_waitcnt lgkmcnt(3)
v_mfma_f32_16x16x32_f16 v[118:121], v[146:149], a[56:59], v[118:121]
s_waitcnt lgkmcnt(2)
v_mfma_f32_16x16x32_f16 v[118:121], v[150:153], a[60:63], v[118:121]
s_waitcnt lgkmcnt(1)
v_mfma_f32_16x16x32_f16 v[114:117], v[154:157], a[56:59], v[114:117]
s_waitcnt lgkmcnt(0)
v_mfma_f32_16x16x32_f16 v[114:117], v[158:161], a[60:63], v[114:117]
v_mfma_f32_16x16x32_f16 v[110:113], v[130:133], a[48:51], v[110:113]
v_mfma_f32_16x16x32_f16 v[110:113], v[134:137], a[52:55], v[110:113]
v_mfma_f32_16x16x32_f16 v[106:109], v[138:141], a[48:51], v[106:109]
v_mfma_f32_16x16x32_f16 v[106:109], v[142:145], a[52:55], v[106:109]
v_mfma_f32_16x16x32_f16 v[102:105], v[146:149], a[48:51], v[102:105]
v_mfma_f32_16x16x32_f16 v[102:105], v[150:153], a[52:55], v[102:105]
v_accvgpr_read_b32 v10, a144
v_accvgpr_read_b32 v11, a145
v_accvgpr_read_b32 v12, a146
v_accvgpr_read_b32 v13, a147
s_nop 1
v_mfma_f32_16x16x32_f16 v[98:101], v[154:157], a[48:51], v[10:13]
v_mfma_f32_16x16x32_f16 v[98:101], v[158:161], a[52:55], v[98:101]
v_cvt_pk_f16_f32 v126, v126, v127
v_cvt_pk_f16_f32 v127, v128, v129
v_cvt_pk_f16_f32 v122, v122, v123
v_cvt_pk_f16_f32 v123, v124, v125
v_cvt_pk_f16_f32 v118, v118, v119
v_cvt_pk_f16_f32 v119, v120, v121
v_cvt_pk_f16_f32 v114, v114, v115
v_cvt_pk_f16_f32 v115, v116, v117
v_cvt_pk_f16_f32 v128, v110, v111
v_cvt_pk_f16_f32 v129, v112, v113
v_cvt_pk_f16_f32 v124, v106, v107
v_cvt_pk_f16_f32 v125, v108, v109
v_cvt_pk_f16_f32 v120, v102, v103
v_cvt_pk_f16_f32 v121, v104, v105
v_cvt_pk_f16_f32 v116, v98, v99
v_cvt_pk_f16_f32 v117, v100, v101
s_waitcnt lgkmcnt(0)
s_barrier
ds_write_b128 v0, v[126:129]
ds_write_b128 v1, v[122:125]
ds_write_b128 v2, v[118:121]
ds_write_b128 v3, v[114:117]
s_waitcnt lgkmcnt(0)
s_barrier
ds_read_b128 v[10:13], v4
ds_read_b128 v[16:19], v4, offset:256
ds_read_b128 v[20:23], v4, offset:128
ds_read_b128 v[26:29], v4, offset:384
s_and_b32 s17, s10, 0xffff
s_mov_b32 s18, s6
s_mov_b32 s19, s7
s_waitcnt lgkmcnt(3)
v_mov_b32_e32 v14, v10
v_mov_b32_e32 v15, v11
s_waitcnt lgkmcnt(2)
buffer_store_dwordx4 v[14:17], v5, s[16:19], 0, offen
s_nop 1
v_mov_b32_e32 v14, v18
v_mov_b32_e32 v15, v19
buffer_store_dwordx4 v[12:15], v6, s[16:19], 0, offen
s_waitcnt lgkmcnt(1)
v_mov_b32_e32 v24, v20
v_mov_b32_e32 v25, v21
s_waitcnt lgkmcnt(0)
buffer_store_dwordx4 v[24:27], v7, s[16:19], 0, offen
s_nop 1
v_mov_b32_e32 v24, v28
v_mov_b32_e32 v25, v29
buffer_store_dwordx4 v[22:25], v8, s[16:19], 0, offen
v_accvgpr_read_b32 v10, a140
v_accvgpr_read_b32 v11, a141
v_accvgpr_read_b32 v12, a142
v_accvgpr_read_b32 v13, a143
s_nop 1
v_mfma_f32_16x16x32_f16 v[94:97], v[130:133], a[40:43], v[10:13]
v_mfma_f32_16x16x32_f16 v[94:97], v[134:137], a[44:47], v[94:97]
s_nop 1
v_accvgpr_read_b32 v10, a136
v_accvgpr_read_b32 v11, a137
v_accvgpr_read_b32 v12, a138
v_accvgpr_read_b32 v13, a139
s_nop 1
v_mfma_f32_16x16x32_f16 v[90:93], v[138:141], a[40:43], v[10:13]
v_mfma_f32_16x16x32_f16 v[90:93], v[142:145], a[44:47], v[90:93]
v_mfma_f32_16x16x32_f16 v[86:89], v[146:149], a[40:43], v[86:89]
v_mfma_f32_16x16x32_f16 v[86:89], v[150:153], a[44:47], v[86:89]
v_mfma_f32_16x16x32_f16 v[82:85], v[154:157], a[40:43], v[82:85]
v_mfma_f32_16x16x32_f16 v[82:85], v[158:161], a[44:47], v[82:85]
v_mfma_f32_16x16x32_f16 v[78:81], v[130:133], a[32:35], v[78:81]
v_mfma_f32_16x16x32_f16 v[78:81], v[134:137], a[36:39], v[78:81]
v_mfma_f32_16x16x32_f16 v[74:77], v[138:141], a[32:35], v[74:77]
v_mfma_f32_16x16x32_f16 v[74:77], v[142:145], a[36:39], v[74:77]
v_mfma_f32_16x16x32_f16 v[70:73], v[146:149], a[32:35], v[70:73]
v_mfma_f32_16x16x32_f16 v[70:73], v[150:153], a[36:39], v[70:73]
v_mfma_f32_16x16x32_f16 v[66:69], v[154:157], a[32:35], v[66:69]
v_mfma_f32_16x16x32_f16 v[66:69], v[158:161], a[36:39], v[66:69]
v_cvt_pk_f16_f32 v94, v94, v95
v_cvt_pk_f16_f32 v95, v96, v97
v_cvt_pk_f16_f32 v90, v90, v91
v_cvt_pk_f16_f32 v91, v92, v93
v_cvt_pk_f16_f32 v86, v86, v87
v_cvt_pk_f16_f32 v87, v88, v89
v_cvt_pk_f16_f32 v82, v82, v83
v_cvt_pk_f16_f32 v83, v84, v85
v_cvt_pk_f16_f32 v96, v78, v79
v_cvt_pk_f16_f32 v97, v80, v81
v_cvt_pk_f16_f32 v92, v74, v75
v_cvt_pk_f16_f32 v93, v76, v77
v_cvt_pk_f16_f32 v88, v70, v71
v_cvt_pk_f16_f32 v89, v72, v73
v_cvt_pk_f16_f32 v84, v66, v67
v_cvt_pk_f16_f32 v85, v68, v69
s_waitcnt lgkmcnt(0)
s_barrier
ds_write_b128 v0, v[94:97]
ds_write_b128 v1, v[90:93]
ds_write_b128 v2, v[86:89]
ds_write_b128 v3, v[82:85]
s_waitcnt lgkmcnt(0)
s_barrier
ds_read_b128 v[10:13], v4
ds_read_b128 v[16:19], v4, offset:256
ds_read_b128 v[20:23], v4, offset:128
ds_read_b128 v[26:29], v4, offset:384
s_and_b32 s13, s3, 0xffff
s_mov_b32 s14, s6
s_mov_b32 s15, s7
s_waitcnt lgkmcnt(3)
v_mov_b32_e32 v14, v10
v_mov_b32_e32 v15, v11
s_waitcnt lgkmcnt(2)
buffer_store_dwordx4 v[14:17], v5, s[12:15], 0, offen
s_nop 1
v_mov_b32_e32 v14, v18
v_mov_b32_e32 v15, v19
buffer_store_dwordx4 v[12:15], v6, s[12:15], 0, offen
s_waitcnt lgkmcnt(1)
v_mov_b32_e32 v24, v20
v_mov_b32_e32 v25, v21
s_waitcnt lgkmcnt(0)
buffer_store_dwordx4 v[24:27], v7, s[12:15], 0, offen
s_nop 1
v_mov_b32_e32 v24, v28
v_mov_b32_e32 v25, v29
buffer_store_dwordx4 v[22:25], v8, s[12:15], 0, offen
v_mfma_f32_16x16x32_f16 v[62:65], v[130:133], a[24:27], v[62:65]
v_mfma_f32_16x16x32_f16 v[62:65], v[134:137], a[28:31], v[62:65]
v_mfma_f32_16x16x32_f16 v[58:61], v[138:141], a[24:27], v[58:61]
v_mfma_f32_16x16x32_f16 v[58:61], v[142:145], a[28:31], v[58:61]
v_mfma_f32_16x16x32_f16 v[54:57], v[146:149], a[24:27], v[42:45]
v_mfma_f32_16x16x32_f16 v[54:57], v[150:153], a[28:31], v[54:57]
v_mfma_f32_16x16x32_f16 v[50:53], v[154:157], a[24:27], v[50:53]
v_mfma_f32_16x16x32_f16 v[50:53], v[158:161], a[28:31], v[50:53]
v_mov_b64_e32 v[48:49], v[40:41]
v_mov_b64_e32 v[46:47], v[38:39]
s_nop 1
v_mfma_f32_16x16x32_f16 v[46:49], v[130:133], a[16:19], v[46:49]
v_mfma_f32_16x16x32_f16 v[46:49], v[134:137], a[20:23], v[46:49]
v_accvgpr_read_b32 v10, a132
v_accvgpr_read_b32 v11, a133
v_accvgpr_read_b32 v12, a134
v_accvgpr_read_b32 v13, a135
s_nop 1
v_mfma_f32_16x16x32_f16 v[42:45], v[138:141], a[16:19], v[10:13]
v_mfma_f32_16x16x32_f16 v[42:45], v[142:145], a[20:23], v[42:45]
s_nop 1
v_accvgpr_read_b32 v10, a128
v_accvgpr_read_b32 v11, a129
v_accvgpr_read_b32 v12, a130
v_accvgpr_read_b32 v13, a131
s_nop 1
v_mfma_f32_16x16x32_f16 v[38:41], v[146:149], a[16:19], v[10:13]
v_mfma_f32_16x16x32_f16 v[38:41], v[150:153], a[20:23], v[38:41]
s_nop 1
v_accvgpr_read_b32 v10, a124
v_accvgpr_read_b32 v11, a125
v_accvgpr_read_b32 v12, a126
v_accvgpr_read_b32 v13, a127
s_nop 1
v_mfma_f32_16x16x32_f16 v[34:37], v[154:157], a[16:19], v[10:13]
v_mfma_f32_16x16x32_f16 v[34:37], v[158:161], a[20:23], v[34:37]
v_cvt_pk_f16_f32 v62, v62, v63
v_cvt_pk_f16_f32 v63, v64, v65
v_cvt_pk_f16_f32 v58, v58, v59
v_cvt_pk_f16_f32 v59, v60, v61
v_cvt_pk_f16_f32 v54, v54, v55
v_cvt_pk_f16_f32 v55, v56, v57
v_cvt_pk_f16_f32 v50, v50, v51
v_cvt_pk_f16_f32 v51, v52, v53
v_cvt_pk_f16_f32 v64, v46, v47
v_cvt_pk_f16_f32 v65, v48, v49
v_cvt_pk_f16_f32 v60, v42, v43
v_cvt_pk_f16_f32 v61, v44, v45
v_cvt_pk_f16_f32 v56, v38, v39
v_cvt_pk_f16_f32 v57, v40, v41
v_cvt_pk_f16_f32 v52, v34, v35
v_cvt_pk_f16_f32 v53, v36, v37
s_waitcnt lgkmcnt(0)
s_barrier
ds_write_b128 v0, v[62:65]
ds_write_b128 v1, v[58:61]
ds_write_b128 v2, v[54:57]
ds_write_b128 v3, v[50:53]
s_waitcnt lgkmcnt(0)
s_barrier
ds_read_b128 v[10:13], v4
ds_read_b128 v[16:19], v4, offset:256
ds_read_b128 v[20:23], v4, offset:128
ds_read_b128 v[26:29], v4, offset:384
s_and_b32 s9, s2, 0xffff
s_mov_b32 s10, s6
s_mov_b32 s11, s7
s_waitcnt lgkmcnt(3)
v_mov_b32_e32 v14, v10
v_mov_b32_e32 v15, v11
s_waitcnt lgkmcnt(2)
buffer_store_dwordx4 v[14:17], v5, s[8:11], 0, offen
s_nop 1
v_mov_b32_e32 v14, v18
v_mov_b32_e32 v15, v19
buffer_store_dwordx4 v[12:15], v6, s[8:11], 0, offen
s_waitcnt lgkmcnt(1)
v_mov_b32_e32 v24, v20
v_mov_b32_e32 v25, v21
s_waitcnt lgkmcnt(0)
buffer_store_dwordx4 v[24:27], v7, s[8:11], 0, offen
s_nop 1
v_mov_b32_e32 v24, v28
v_mov_b32_e32 v25, v29
buffer_store_dwordx4 v[22:25], v8, s[8:11], 0, offen
v_accvgpr_read_b32 v10, a120
v_accvgpr_read_b32 v11, a121
v_accvgpr_read_b32 v12, a122
v_accvgpr_read_b32 v13, a123
s_nop 1
v_mfma_f32_16x16x32_f16 v[30:33], v[130:133], a[8:11], v[10:13]
v_mfma_f32_16x16x32_f16 v[30:33], v[134:137], a[12:15], v[30:33]
s_nop 1
v_accvgpr_read_b32 v10, a250
v_accvgpr_read_b32 v11, a251
v_accvgpr_read_b32 v12, a252
v_accvgpr_read_b32 v13, a253
s_nop 1
v_mfma_f32_16x16x32_f16 v[26:29], v[138:141], a[8:11], v[10:13]
v_mfma_f32_16x16x32_f16 v[26:29], v[142:145], a[12:15], v[26:29]
s_nop 1
v_accvgpr_read_b32 v10, a246
v_accvgpr_read_b32 v11, a247
v_accvgpr_read_b32 v12, a248
v_accvgpr_read_b32 v13, a249
s_nop 1
v_mfma_f32_16x16x32_f16 v[22:25], v[146:149], a[8:11], v[10:13]
v_mfma_f32_16x16x32_f16 v[22:25], v[150:153], a[12:15], v[22:25]
s_nop 1
v_accvgpr_read_b32 v10, a242
v_accvgpr_read_b32 v11, a243
v_accvgpr_read_b32 v12, a244
v_accvgpr_read_b32 v13, a245
s_nop 1
v_mfma_f32_16x16x32_f16 v[18:21], v[154:157], a[8:11], v[10:13]
v_mfma_f32_16x16x32_f16 v[18:21], v[158:161], a[12:15], v[18:21]
s_nop 1
v_accvgpr_read_b32 v10, a206
v_accvgpr_read_b32 v11, a207
v_accvgpr_read_b32 v12, a208
v_accvgpr_read_b32 v13, a209
s_nop 1
v_mfma_f32_16x16x32_f16 v[14:17], v[130:133], a[0:3], v[10:13]
v_mfma_f32_16x16x32_f16 v[14:17], v[134:137], a[4:7], v[14:17]
s_nop 1
v_accvgpr_read_b32 v10, a222
v_accvgpr_read_b32 v11, a223
v_accvgpr_read_b32 v12, a224
v_accvgpr_read_b32 v13, a225
s_nop 1
v_mfma_f32_16x16x32_f16 v[10:13], v[138:141], a[0:3], v[10:13]
v_mfma_f32_16x16x32_f16 v[10:13], v[142:145], a[4:7], v[10:13]
v_accvgpr_read_b32 v34, a218
v_accvgpr_read_b32 v35, a219
v_accvgpr_read_b32 v36, a220
v_accvgpr_read_b32 v37, a221
s_nop 1
v_mfma_f32_16x16x32_f16 v[34:37], v[146:149], a[0:3], v[34:37]
v_mfma_f32_16x16x32_f16 v[34:37], v[150:153], a[4:7], v[34:37]
v_mfma_f32_16x16x32_f16 v[38:41], v[154:157], a[0:3], v[250:253]
v_mfma_f32_16x16x32_f16 v[38:41], v[158:161], a[4:7], v[38:41]
v_cvt_pk_f16_f32 v30, v30, v31
v_cvt_pk_f16_f32 v31, v32, v33
v_cvt_pk_f16_f32 v26, v26, v27
v_cvt_pk_f16_f32 v27, v28, v29
v_cvt_pk_f16_f32 v22, v22, v23
v_cvt_pk_f16_f32 v23, v24, v25
v_cvt_pk_f16_f32 v18, v18, v19
v_cvt_pk_f16_f32 v19, v20, v21
v_cvt_pk_f16_f32 v32, v14, v15
v_cvt_pk_f16_f32 v33, v16, v17
v_cvt_pk_f16_f32 v28, v10, v11
v_cvt_pk_f16_f32 v29, v12, v13
v_cvt_pk_f16_f32 v24, v34, v35
v_cvt_pk_f16_f32 v25, v36, v37
v_cvt_pk_f16_f32 v20, v38, v39
v_cvt_pk_f16_f32 v21, v40, v41
s_waitcnt lgkmcnt(0)
s_barrier
ds_write_b128 v0, v[30:33]
ds_write_b128 v1, v[26:29]
ds_write_b128 v2, v[22:25]
ds_write_b128 v3, v[18:21]
s_waitcnt lgkmcnt(0)
s_barrier
ds_read_b128 v[0:3], v4
ds_read_b128 v[12:15], v4, offset:256
ds_read_b128 v[16:19], v4, offset:128
ds_read_b128 v[22:25], v4, offset:384
s_and_b32 s1, s1, 0xffff
s_mov_b32 s2, s6
s_mov_b32 s3, s7
s_waitcnt lgkmcnt(3)
v_mov_b32_e32 v10, v0
v_mov_b32_e32 v11, v1
s_waitcnt lgkmcnt(2)
buffer_store_dwordx4 v[10:13], v5, s[0:3], 0, offen
v_mov_b32_e32 v4, v14
v_mov_b32_e32 v5, v15
buffer_store_dwordx4 v[2:5], v6, s[0:3], 0, offen
s_waitcnt lgkmcnt(1)
v_mov_b32_e32 v20, v16
v_mov_b32_e32 v21, v17
s_waitcnt lgkmcnt(0)
buffer_store_dwordx4 v[20:23], v7, s[0:3], 0, offen
s_nop 1
v_mov_b32_e32 v20, v24
v_mov_b32_e32 v21, v25
buffer_store_dwordx4 v[18:21], v8, s[0:3], 0, offen
s_endpgm
	s_endpgm
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel v8_beyond_hotloop
		.amdhsa_group_segment_fixed_size 0
		.amdhsa_private_segment_fixed_size 0
		.amdhsa_kernarg_size 64
		.amdhsa_user_sgpr_count 16
		.amdhsa_user_sgpr_dispatch_ptr 0
		.amdhsa_user_sgpr_queue_ptr 0
		.amdhsa_user_sgpr_kernarg_segment_ptr 1
		.amdhsa_user_sgpr_dispatch_id 0
		.amdhsa_user_sgpr_kernarg_preload_length 14
		.amdhsa_user_sgpr_kernarg_preload_offset 0
		.amdhsa_user_sgpr_private_segment_size 0
		.amdhsa_uses_dynamic_stack 0
		.amdhsa_enable_private_segment 0
		.amdhsa_system_sgpr_workgroup_id_x 1
		.amdhsa_system_sgpr_workgroup_id_y 0
		.amdhsa_system_sgpr_workgroup_id_z 0
		.amdhsa_system_sgpr_workgroup_info 0
		.amdhsa_system_vgpr_workitem_id 0
		.amdhsa_next_free_vgpr 512
		.amdhsa_next_free_sgpr 60
		.amdhsa_accum_offset 256
		.amdhsa_reserve_vcc 1
		.amdhsa_reserve_xnack_mask 1
		.amdhsa_float_round_mode_32 0
		.amdhsa_float_round_mode_16_64 0
		.amdhsa_float_denorm_mode_32 3
		.amdhsa_float_denorm_mode_16_64 3
		.amdhsa_dx10_clamp 1
		.amdhsa_ieee_mode 1
		.amdhsa_fp16_overflow 0
		.amdhsa_tg_split 0
		.amdhsa_exception_fp_ieee_invalid_op 0
		.amdhsa_exception_fp_denorm_src 0
		.amdhsa_exception_fp_ieee_div_zero 0
		.amdhsa_exception_fp_ieee_overflow 0
		.amdhsa_exception_fp_ieee_underflow 0
		.amdhsa_exception_fp_ieee_inexact 0
		.amdhsa_exception_int_div_zero 0
	.end_amdhsa_kernel
	.text
.Lfunc_end0:
	.size	v8_beyond_hotloop, .Lfunc_end0-v8_beyond_hotloop
	.cfi_endproc
                                        ; -- End function
	.set v8_beyond_hotloop.num_vgpr, 256
	.set v8_beyond_hotloop.num_agpr, 254
	.set v8_beyond_hotloop.numbered_sgpr, 60
	.set v8_beyond_hotloop.num_named_barrier, 0
	.set v8_beyond_hotloop.private_seg_size, 0
	.set v8_beyond_hotloop.uses_vcc, 1
	.set v8_beyond_hotloop.uses_flat_scratch, 0
	.set v8_beyond_hotloop.has_dyn_sized_stack, 0
	.set v8_beyond_hotloop.has_recursion, 0
	.set v8_beyond_hotloop.has_indirect_call, 0
	.section	.AMDGPU.csdata,"",@progbits
; Kernel info:
; codeLenInByte = 14236
; TotalNumSgprs: 66
; NumVgprs: 256
; NumAgprs: 254
; TotalNumVgprs: 510
; ScratchSize: 0
; MemoryBound: 0
; FloatMode: 240
; IeeeMode: 1
; LDSByteSize: 0 bytes/workgroup (compile time only)
; SGPRBlocks: 8
; VGPRBlocks: 63
; NumSGPRsForWavesPerEU: 66
; NumVGPRsForWavesPerEU: 510
; AccumOffset: 256
; Occupancy: 1
; WaveLimiterHint : 0
; COMPUTE_PGM_RSRC2:SCRATCH_EN: 0
; COMPUTE_PGM_RSRC2:USER_SGPR: 16
; COMPUTE_PGM_RSRC2:TRAP_HANDLER: 0
; COMPUTE_PGM_RSRC2:TGID_X_EN: 1
; COMPUTE_PGM_RSRC2:TGID_Y_EN: 0
; COMPUTE_PGM_RSRC2:TGID_Z_EN: 0
; COMPUTE_PGM_RSRC2:TIDIG_COMP_CNT: 0
; COMPUTE_PGM_RSRC3_GFX90A:ACCUM_OFFSET: 63
; COMPUTE_PGM_RSRC3_GFX90A:TG_SPLIT: 0
	.text
	.p2alignl 6, 3212836864
	.fill 256, 4, 3212836864
	.section	.AMDGPU.gpr_maximums,"",@progbits
	.set amdgpu.max_num_vgpr, 0
	.set amdgpu.max_num_agpr, 0
	.set amdgpu.max_num_sgpr, 0
	.set amdgpu.max_num_named_barrier, 0
	.text
	.section	.debug_abbrev,"",@progbits
	.byte	1                               ; Abbreviation Code
	.byte	17                              ; DW_TAG_compile_unit
	.byte	1                               ; DW_CHILDREN_yes
	.byte	37                              ; DW_AT_producer
	.byte	14                              ; DW_FORM_strp
	.byte	19                              ; DW_AT_language
	.byte	5                               ; DW_FORM_data2
	.byte	3                               ; DW_AT_name
	.byte	14                              ; DW_FORM_strp
	.byte	16                              ; DW_AT_stmt_list
	.byte	23                              ; DW_FORM_sec_offset
	.byte	27                              ; DW_AT_comp_dir
	.byte	14                              ; DW_FORM_strp
	.byte	17                              ; DW_AT_low_pc
	.byte	1                               ; DW_FORM_addr
	.byte	18                              ; DW_AT_high_pc
	.byte	6                               ; DW_FORM_data4
	.byte	0                               ; EOM(1)
	.byte	0                               ; EOM(2)
	.byte	2                               ; Abbreviation Code
	.byte	46                              ; DW_TAG_subprogram
	.byte	0                               ; DW_CHILDREN_no
	.byte	3                               ; DW_AT_name
	.byte	14                              ; DW_FORM_strp
	.byte	32                              ; DW_AT_inline
	.byte	11                              ; DW_FORM_data1
	.byte	0                               ; EOM(1)
	.byte	0                               ; EOM(2)
	.byte	3                               ; Abbreviation Code
	.byte	46                              ; DW_TAG_subprogram
	.byte	1                               ; DW_CHILDREN_yes
	.byte	17                              ; DW_AT_low_pc
	.byte	1                               ; DW_FORM_addr
	.byte	18                              ; DW_AT_high_pc
	.byte	6                               ; DW_FORM_data4
	.byte	49                              ; DW_AT_abstract_origin
	.byte	19                              ; DW_FORM_ref4
	.byte	0                               ; EOM(1)
	.byte	0                               ; EOM(2)
	.byte	4                               ; Abbreviation Code
	.byte	29                              ; DW_TAG_inlined_subroutine
	.byte	1                               ; DW_CHILDREN_yes
	.byte	49                              ; DW_AT_abstract_origin
	.byte	19                              ; DW_FORM_ref4
	.byte	17                              ; DW_AT_low_pc
	.byte	1                               ; DW_FORM_addr
	.byte	18                              ; DW_AT_high_pc
	.byte	6                               ; DW_FORM_data4
	.byte	88                              ; DW_AT_call_file
	.byte	11                              ; DW_FORM_data1
	.byte	89                              ; DW_AT_call_line
	.byte	11                              ; DW_FORM_data1
	.byte	87                              ; DW_AT_call_column
	.byte	11                              ; DW_FORM_data1
	.byte	0                               ; EOM(1)
	.byte	0                               ; EOM(2)
	.byte	5                               ; Abbreviation Code
	.byte	29                              ; DW_TAG_inlined_subroutine
	.byte	0                               ; DW_CHILDREN_no
	.byte	49                              ; DW_AT_abstract_origin
	.byte	19                              ; DW_FORM_ref4
	.byte	17                              ; DW_AT_low_pc
	.byte	1                               ; DW_FORM_addr
	.byte	18                              ; DW_AT_high_pc
	.byte	6                               ; DW_FORM_data4
	.byte	88                              ; DW_AT_call_file
	.byte	11                              ; DW_FORM_data1
	.byte	89                              ; DW_AT_call_line
	.byte	11                              ; DW_FORM_data1
	.byte	87                              ; DW_AT_call_column
	.byte	11                              ; DW_FORM_data1
	.byte	0                               ; EOM(1)
	.byte	0                               ; EOM(2)
	.byte	0                               ; EOM(3)
	.section	.debug_str,"MS",@progbits,1
.Linfo_string0:
	.asciz	"triton"                        ; string offset=0 ; triton
.Linfo_string1:
	.asciz	"matmul_kernel.py"              ; string offset=7 ; matmul_kernel.py
.Linfo_string2:
	.asciz	"/var/lib/jenkins/gfx9-gluon-tutorials/kernels/gemm/a16w16/v8_beyond_hotloop" ; string offset=24 ; /var/lib/jenkins/gfx9-gluon-tutorials/kernels/gemm/a16w16/v8_beyond_hotloop
.Linfo_string3:
	.asciz	"v8_beyond_hotloop"             ; string offset=100 ; v8_beyond_hotloop
	.section	".note.GNU-stack","",@progbits
	.amdgpu_metadata
---
amdhsa.kernels:
  - .agpr_count:     254
    .args:
      - .address_space:  global
        .offset:         0
        .size:           8
        .value_kind:     global_buffer
      - .address_space:  global
        .offset:         8
        .size:           8
        .value_kind:     global_buffer
      - .address_space:  global
        .offset:         16
        .size:           8
        .value_kind:     global_buffer
      - .offset:         24
        .size:           4
        .value_kind:     by_value
      - .offset:         28
        .size:           4
        .value_kind:     by_value
      - .offset:         32
        .size:           4
        .value_kind:     by_value
      - .offset:         36
        .size:           4
        .value_kind:     by_value
      - .offset:         40
        .size:           4
        .value_kind:     by_value
      - .offset:         44
        .size:           4
        .value_kind:     by_value
      - .address_space:  global
        .offset:         48
        .size:           8
        .value_kind:     global_buffer
      - .address_space:  global
        .offset:         56
        .size:           8
        .value_kind:     global_buffer
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 64
    .max_flat_workgroup_size: 256
    .name:           v8_beyond_hotloop
    .private_segment_fixed_size: 0
    .sgpr_count:     66
    .sgpr_spill_count: 0
    .symbol:         v8_beyond_hotloop.kd
    .uniform_work_group_size: 1
    .uses_dynamic_stack: false
    .vgpr_count: 512
    .vgpr_spill_count: 0
    .wavefront_size: 64
amdhsa.target:   amdgcn-amd-amdhsa--gfx950
amdhsa.version:
  - 1
  - 2
...

	.end_amdgpu_metadata
	.section	.debug_line,"",@progbits
.Lline_table_start0: