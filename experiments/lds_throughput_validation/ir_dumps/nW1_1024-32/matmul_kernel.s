	.amdgcn_target "amdgcn-amd-amdhsa--gfx950"
	.amdhsa_code_object_version 5
	.text
	.globl	matmul_kernel                   ; -- Begin function matmul_kernel
	.p2align	8
	.type	matmul_kernel,@function
matmul_kernel:                          ; @matmul_kernel
.Lfunc_begin0:
	.cfi_sections .debug_frame
	.cfi_startproc
; %bb.1:
	.file	1 "/root/OAI-triton/study_lds" "play_lds.py"
	s_load_dwordx2 s[2:3], s[0:1], 0x0
	s_load_dwordx8 s[4:11], s[0:1], 0x8
	s_load_dwordx4 s[12:15], s[0:1], 0x28
	s_waitcnt lgkmcnt(0)
	s_branch .LBB0_0
	.p2align	8
; %bb.2:
.LBB0_0:
	s_ashr_i32 s0, s16, 31
	s_lshr_b32 s0, s0, 29
	s_add_i32 s0, s16, s0
	s_ashr_i32 s0, s0, 3
	s_lshl_b32 s1, s16, 9
	s_mulk_i32 s0, 0xf001
	s_add_i32 s0, s0, s1
	.file	2 "/root/OAI-triton/python/triton/language" "standard.py"
	s_add_i32 s1, s9, 0xff
	s_ashr_i32 s9, s1, 31
	s_lshr_b32 s9, s9, 24
	s_add_i32 s1, s1, s9
	s_ashr_i32 s1, s1, 8
	s_add_i32 s8, s8, 15
	s_ashr_i32 s9, s8, 31
	s_lshr_b32 s9, s9, 28
	s_add_i32 s8, s8, s9
	s_ashr_i32 s8, s8, 4
	s_lshl_b32 s9, s1, 2
	s_xor_b32 s1, s0, s1
	s_ashr_i32 s1, s1, 31
	s_abs_i32 s13, s0
	s_abs_i32 s14, s9
	v_cvt_f32_u32_e32 v1, s14
	v_rcp_iflag_f32_e32 v1, v1
	s_nop 0
	v_mul_f32_e32 v1, 0x4f7ffffe, v1
	v_cvt_u32_f32_e32 v1, v1
	s_sub_i32 s15, 0, s14
	v_readfirstlane_b32 s16, v1
	s_mul_i32 s15, s15, s16
	s_mul_hi_u32 s15, s16, s15
	s_add_i32 s16, s16, s15
	s_mul_hi_u32 s15, s13, s16
	s_mul_i32 s16, s15, s14
	s_sub_i32 s13, s13, s16
	s_add_i32 s16, s15, 1
	s_sub_i32 s17, s13, s14
	s_cmp_ge_u32 s13, s14
	s_cselect_b32 s15, s16, s15
	s_cselect_b32 s13, s17, s13
	s_add_i32 s16, s15, 1
	s_cmp_ge_u32 s13, s14
	s_cselect_b32 s13, s16, s15
	s_xor_b32 s13, s13, s1
	s_sub_i32 s1, s13, s1
	s_lshl_b32 s13, s1, 2
	s_sub_i32 s8, s8, s13
	s_min_i32 s8, s8, 4
	s_mul_i32 s1, s1, s9
	s_sub_i32 s0, s0, s1
	s_xor_b32 s1, s0, s8
	s_ashr_i32 s1, s1, 31
	s_abs_i32 s9, s0
	s_abs_i32 s14, s8
	v_cvt_f32_u32_e32 v1, s14
	v_rcp_iflag_f32_e32 v1, v1
	s_nop 0
	v_mul_f32_e32 v1, 0x4f7ffffe, v1
	v_cvt_u32_f32_e32 v1, v1
	s_sub_i32 s15, 0, s14
	v_readfirstlane_b32 s16, v1
	s_mul_i32 s15, s15, s16
	s_mul_hi_u32 s15, s16, s15
	s_add_i32 s16, s16, s15
	s_mul_hi_u32 s15, s9, s16
	s_mul_i32 s16, s15, s14
	s_sub_i32 s9, s9, s16
	s_add_i32 s16, s15, 1
	s_sub_i32 s17, s9, s14
	s_cmp_ge_u32 s9, s14
	s_cselect_b32 s15, s16, s15
	s_cselect_b32 s9, s17, s9
	s_add_i32 s16, s15, 1
	s_cmp_ge_u32 s9, s14
	s_cselect_b32 s9, s16, s15
	s_xor_b32 s9, s9, s1
	s_sub_i32 s1, s9, s1
	s_mul_i32 s8, s1, s8
	s_sub_i32 s0, s0, s8
	s_add_i32 s0, s0, s13
	v_and_b32_e32 v64, 15, v0
	v_and_b32_e32 v65, 48, v0
	v_lshlrev_b32_e32 v2, 1, v65
	v_lshlrev_b32_e32 v1, 1, v0
	v_and_b32_e32 v1, 0x70, v1
	v_lshlrev_b32_e32 v0, 4, v0
	v_and_b32_e32 v0, 0x70, v0
	s_lshl_b32 s13, s0, 4
	s_mul_i32 s0, s13, s10
	s_ashr_i32 s8, s0, 31
	s_add_u32 s0, s2, s0
	s_addc_u32 s9, s3, s8
	s_lshl_b32 s14, s1, 8
	s_mul_i32 s1, s14, s11
	s_ashr_i32 s2, s1, 31
	s_add_u32 s8, s4, s1
	s_addc_u32 s4, s5, s2
	v_mad_u64_u32 v[4:5], s[2:3], s10, v64, v[2:3]
	s_mul_i32 s1, s11, 0x71
	v_mad_u64_u32 v[0:1], s[2:3], s11, v1, v[0:1]
	v_add_u32_e32 v1, s11, v0
	v_add_u32_e32 v3, s11, v1
	v_add_u32_e32 v5, s11, v3
	v_add_u32_e32 v6, s11, v5
	v_add_u32_e32 v7, s11, v6
	v_add_u32_e32 v8, s11, v7
	v_add_u32_e32 v9, s11, v8
	v_add_u32_e32 v10, s11, v9
	v_add_u32_e32 v11, s11, v10
	v_add_u32_e32 v12, s11, v11
	v_add_u32_e32 v13, s11, v12
	v_add_u32_e32 v14, s11, v13
	v_add_u32_e32 v15, s11, v14
	v_add_u32_e32 v16, s11, v15
	v_add_u32_e32 v17, s11, v16
	v_add_u32_e32 v18, s1, v17
	v_add_u32_e32 v19, s11, v18
	v_add_u32_e32 v20, s11, v19
	v_add_u32_e32 v21, s11, v20
	v_add_u32_e32 v22, s11, v21
	v_add_u32_e32 v23, s11, v22
	v_add_u32_e32 v24, s11, v23
	v_add_u32_e32 v25, s11, v24
	v_add_u32_e32 v26, s11, v25
	v_add_u32_e32 v27, s11, v26
	v_add_u32_e32 v28, s11, v27
	v_add_u32_e32 v29, s11, v28
	v_add_u32_e32 v30, s11, v29
	v_add_u32_e32 v31, s11, v30
	v_add_u32_e32 v32, s11, v31
	v_add_u32_e32 v33, s11, v32
	s_and_b32 s1, s9, 0xffff
	s_mov_b32 s3, 0x27000
	s_mov_b32 s2, 0x7ffffffe
	buffer_load_dwordx4 v[66:69], v4, s[0:3], 0 offen
	buffer_load_dwordx4 v[70:73], v4, s[0:3], 0 offen offset:16
	s_and_b32 s9, s4, 0xffff
	s_mov_b32 s10, s2
	s_mov_b32 s11, s3
	s_mov_b32 m0, 0
	s_nop 0
	buffer_load_dwordx4 v0, s[8:11], 0 offen lds
	s_add_i32 m0, 0, 0x420
	s_nop 0
	buffer_load_dwordx4 v1, s[8:11], 0 offen lds
	s_add_i32 m0, 0, 0x840
	s_nop 0
	buffer_load_dwordx4 v3, s[8:11], 0 offen lds
	s_add_i32 m0, 0, 0xc60
	s_nop 0
	buffer_load_dwordx4 v5, s[8:11], 0 offen lds
	s_add_i32 m0, 0, 0x1080
	s_nop 0
	buffer_load_dwordx4 v6, s[8:11], 0 offen lds
	s_add_i32 m0, 0, 0x14a0
	s_nop 0
	buffer_load_dwordx4 v7, s[8:11], 0 offen lds
	s_add_i32 m0, 0, 0x18c0
	s_nop 0
	buffer_load_dwordx4 v8, s[8:11], 0 offen lds
	s_add_i32 m0, 0, 0x1ce0
	s_nop 0
	buffer_load_dwordx4 v9, s[8:11], 0 offen lds
	s_add_i32 m0, 0, 0x2100
	s_nop 0
	buffer_load_dwordx4 v10, s[8:11], 0 offen lds
	s_add_i32 m0, 0, 0x2520
	s_nop 0
	buffer_load_dwordx4 v11, s[8:11], 0 offen lds
	s_add_i32 m0, 0, 0x2940
	s_nop 0
	buffer_load_dwordx4 v12, s[8:11], 0 offen lds
	s_add_i32 m0, 0, 0x2d60
	s_nop 0
	buffer_load_dwordx4 v13, s[8:11], 0 offen lds
	s_add_i32 m0, 0, 0x3180
	s_nop 0
	buffer_load_dwordx4 v14, s[8:11], 0 offen lds
	s_add_i32 m0, 0, 0x35a0
	s_nop 0
	buffer_load_dwordx4 v15, s[8:11], 0 offen lds
	s_add_i32 m0, 0, 0x39c0
	s_nop 0
	buffer_load_dwordx4 v16, s[8:11], 0 offen lds
	s_add_i32 m0, 0, 0x3de0
	s_nop 0
	buffer_load_dwordx4 v17, s[8:11], 0 offen lds
	s_add_i32 m0, 0, 0x4200
	s_nop 0
	buffer_load_dwordx4 v18, s[8:11], 0 offen lds
	s_add_i32 m0, 0, 0x4620
	s_nop 0
	buffer_load_dwordx4 v19, s[8:11], 0 offen lds
	s_add_i32 m0, 0, 0x4a40
	s_nop 0
	buffer_load_dwordx4 v20, s[8:11], 0 offen lds
	s_add_i32 m0, 0, 0x4e60
	s_nop 0
	buffer_load_dwordx4 v21, s[8:11], 0 offen lds
	s_add_i32 m0, 0, 0x5280
	s_nop 0
	buffer_load_dwordx4 v22, s[8:11], 0 offen lds
	s_add_i32 m0, 0, 0x56a0
	s_nop 0
	buffer_load_dwordx4 v23, s[8:11], 0 offen lds
	s_add_i32 m0, 0, 0x5ac0
	s_nop 0
	buffer_load_dwordx4 v24, s[8:11], 0 offen lds
	s_add_i32 m0, 0, 0x5ee0
	s_nop 0
	buffer_load_dwordx4 v25, s[8:11], 0 offen lds
	s_add_i32 m0, 0, 0x6300
	s_nop 0
	buffer_load_dwordx4 v26, s[8:11], 0 offen lds
	s_add_i32 m0, 0, 0x6720
	s_nop 0
	buffer_load_dwordx4 v27, s[8:11], 0 offen lds
	s_add_i32 m0, 0, 0x6b40
	s_nop 0
	buffer_load_dwordx4 v28, s[8:11], 0 offen lds
	s_add_i32 m0, 0, 0x6f60
	s_nop 0
	buffer_load_dwordx4 v29, s[8:11], 0 offen lds
	s_add_i32 m0, 0, 0x7380
	s_nop 0
	buffer_load_dwordx4 v30, s[8:11], 0 offen lds
	s_add_i32 m0, 0, 0x77a0
	s_nop 0
	buffer_load_dwordx4 v31, s[8:11], 0 offen lds
	s_add_i32 m0, 0, 0x7bc0
	s_nop 0
	buffer_load_dwordx4 v32, s[8:11], 0 offen lds
	s_add_i32 m0, 0, 0x7fe0
	s_nop 0
	buffer_load_dwordx4 v33, s[8:11], 0 offen lds
	s_waitcnt vmcnt(0) lgkmcnt(0)
	; wave barrier
	v_lshlrev_b32_e32 v0, 5, v64
	v_lshl_add_u32 v1, v64, 10, 0
	v_add3_u32 v0, v1, v2, v0
	ds_read_b128 v[2:5], v0
	ds_read_b128 v[6:9], v0 offset:16
	ds_read_b128 v[14:17], v0 offset:144
	ds_read_b128 v[10:13], v0 offset:128
	ds_read_b128 v[22:25], v0 offset:272
	ds_read_b128 v[18:21], v0 offset:256
	ds_read_b128 v[30:33], v0 offset:400
	ds_read_b128 v[26:29], v0 offset:384
	ds_read_b128 v[38:41], v0 offset:528
	ds_read_b128 v[34:37], v0 offset:512
	ds_read_b128 v[46:49], v0 offset:656
	ds_read_b128 v[42:45], v0 offset:640
	ds_read_b128 v[54:57], v0 offset:784
	ds_read_b128 v[50:53], v0 offset:768
	ds_read_b128 v[78:81], v0 offset:912
	ds_read_b128 v[74:77], v0 offset:896
	ds_read_b128 v[86:89], v0 offset:16912
	ds_read_b128 v[82:85], v0 offset:16896
	ds_read_b128 v[94:97], v0 offset:17040
	ds_read_b128 v[90:93], v0 offset:17024
	ds_read_b128 v[102:105], v0 offset:17168
	ds_read_b128 v[98:101], v0 offset:17152
	ds_read_b128 v[110:113], v0 offset:17296
	ds_read_b128 v[106:109], v0 offset:17280
	ds_read_b128 v[118:121], v0 offset:17424
	ds_read_b128 v[114:117], v0 offset:17408
	ds_read_b128 v[126:129], v0 offset:17552
	ds_read_b128 v[122:125], v0 offset:17536
	ds_read_b128 v[134:137], v0 offset:17680
	ds_read_b128 v[130:133], v0 offset:17664
	ds_read_b128 v[142:145], v0 offset:17808
	ds_read_b128 v[138:141], v0 offset:17792
	v_mov_b32_e32 v60, 0x7f
	s_waitcnt lgkmcnt(14)
	s_nop 0
	v_mfma_scale_f32_16x16x128_f8f6f4 v[0:3], v[2:9], v[66:73], 0, v60, v60 op_sel_hi:[0,0,0] cbsz:1 blgp:1
	v_mfma_scale_f32_16x16x128_f8f6f4 v[4:7], v[10:17], v[66:73], 0, v60, v60 op_sel_hi:[0,0,0] cbsz:1 blgp:1
	v_mfma_scale_f32_16x16x128_f8f6f4 v[8:11], v[18:25], v[66:73], 0, v60, v60 op_sel_hi:[0,0,0] cbsz:1 blgp:1
	v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[26:33], v[66:73], 0, v60, v60 op_sel_hi:[0,0,0] cbsz:1 blgp:1
	v_mfma_scale_f32_16x16x128_f8f6f4 v[16:19], v[34:41], v[66:73], 0, v60, v60 op_sel_hi:[0,0,0] cbsz:1 blgp:1
	v_mfma_scale_f32_16x16x128_f8f6f4 v[20:23], v[42:49], v[66:73], 0, v60, v60 op_sel_hi:[0,0,0] cbsz:1 blgp:1
	v_mfma_scale_f32_16x16x128_f8f6f4 v[24:27], v[50:57], v[66:73], 0, v60, v60 op_sel_hi:[0,0,0] cbsz:1 blgp:1
	v_mfma_scale_f32_16x16x128_f8f6f4 v[28:31], v[74:81], v[66:73], 0, v60, v60 op_sel_hi:[0,0,0] cbsz:1 blgp:1
	v_mfma_scale_f32_16x16x128_f8f6f4 v[32:35], v[82:89], v[66:73], 0, v60, v60 op_sel_hi:[0,0,0] cbsz:1 blgp:1
	s_waitcnt lgkmcnt(12)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[36:39], v[90:97], v[66:73], 0, v60, v60 op_sel_hi:[0,0,0] cbsz:1 blgp:1
	s_waitcnt lgkmcnt(10)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[40:43], v[98:105], v[66:73], 0, v60, v60 op_sel_hi:[0,0,0] cbsz:1 blgp:1
	s_waitcnt lgkmcnt(8)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[44:47], v[106:113], v[66:73], 0, v60, v60 op_sel_hi:[0,0,0] cbsz:1 blgp:1
	s_waitcnt lgkmcnt(6)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[48:51], v[114:121], v[66:73], 0, v60, v60 op_sel_hi:[0,0,0] cbsz:1 blgp:1
	s_waitcnt lgkmcnt(4)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[52:55], v[122:129], v[66:73], 0, v60, v60 op_sel_hi:[0,0,0] cbsz:1 blgp:1
	s_waitcnt lgkmcnt(2)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[56:59], v[130:137], v[66:73], 0, v60, v60 op_sel_hi:[0,0,0] cbsz:1 blgp:1
	s_waitcnt lgkmcnt(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[60:63], v[138:145], v[66:73], 0, v60, v60 op_sel_hi:[0,0,0] cbsz:1 blgp:1
	v_cvt_pk_f16_f32 v0, v0, v1
	v_cvt_pk_f16_f32 v1, v2, v3
	v_cvt_pk_f16_f32 v2, v4, v5
	v_cvt_pk_f16_f32 v3, v6, v7
	v_cvt_pk_f16_f32 v4, v8, v9
	v_cvt_pk_f16_f32 v5, v10, v11
	v_cvt_pk_f16_f32 v6, v12, v13
	v_cvt_pk_f16_f32 v7, v14, v15
	v_cvt_pk_f16_f32 v8, v16, v17
	v_cvt_pk_f16_f32 v9, v18, v19
	v_cvt_pk_f16_f32 v10, v20, v21
	v_cvt_pk_f16_f32 v11, v22, v23
	v_cvt_pk_f16_f32 v12, v24, v25
	v_cvt_pk_f16_f32 v13, v26, v27
	v_cvt_pk_f16_f32 v14, v28, v29
	v_cvt_pk_f16_f32 v15, v30, v31
	v_cvt_pk_f16_f32 v16, v32, v33
	v_cvt_pk_f16_f32 v17, v34, v35
	v_cvt_pk_f16_f32 v18, v36, v37
	v_cvt_pk_f16_f32 v19, v38, v39
	v_cvt_pk_f16_f32 v20, v40, v41
	v_cvt_pk_f16_f32 v21, v42, v43
	v_cvt_pk_f16_f32 v22, v44, v45
	v_cvt_pk_f16_f32 v23, v46, v47
	v_cvt_pk_f16_f32 v24, v48, v49
	v_cvt_pk_f16_f32 v25, v50, v51
	v_cvt_pk_f16_f32 v26, v52, v53
	v_cvt_pk_f16_f32 v27, v54, v55
	v_cvt_pk_f16_f32 v28, v56, v57
	v_cvt_pk_f16_f32 v29, v58, v59
	v_cvt_pk_f16_f32 v30, v60, v61
	v_cvt_pk_f16_f32 v31, v62, v63
	v_lshrrev_b32_e32 v32, 2, v65
	s_mul_i32 s0, s13, s12
	s_ashr_i32 s1, s0, 31
	s_lshl_b64 s[0:1], s[0:1], 1
	s_add_u32 s4, s6, s0
	s_addc_u32 s5, s7, s1
	s_ashr_i32 s15, s14, 31
	s_lshl_b64 s[0:1], s[14:15], 1
	s_add_u32 s0, s4, s0
	s_addc_u32 s1, s5, s1
	v_mul_lo_u32 v33, s12, v64
	s_and_b32 s1, s1, 0xffff
	v_add_lshl_u32 v32, v33, v32, 1
	buffer_store_dwordx2 v[0:1], v32, s[0:3], 0 offen
	buffer_store_dwordx2 v[2:3], v32, s[0:3], 0 offen offset:32
	buffer_store_dwordx2 v[4:5], v32, s[0:3], 0 offen offset:64
	buffer_store_dwordx2 v[6:7], v32, s[0:3], 0 offen offset:96
	buffer_store_dwordx2 v[8:9], v32, s[0:3], 0 offen offset:128
	buffer_store_dwordx2 v[10:11], v32, s[0:3], 0 offen offset:160
	buffer_store_dwordx2 v[12:13], v32, s[0:3], 0 offen offset:192
	buffer_store_dwordx2 v[14:15], v32, s[0:3], 0 offen offset:224
	buffer_store_dwordx2 v[16:17], v32, s[0:3], 0 offen offset:256
	buffer_store_dwordx2 v[18:19], v32, s[0:3], 0 offen offset:288
	buffer_store_dwordx2 v[20:21], v32, s[0:3], 0 offen offset:320
	buffer_store_dwordx2 v[22:23], v32, s[0:3], 0 offen offset:352
	buffer_store_dwordx2 v[24:25], v32, s[0:3], 0 offen offset:384
	buffer_store_dwordx2 v[26:27], v32, s[0:3], 0 offen offset:416
	buffer_store_dwordx2 v[28:29], v32, s[0:3], 0 offen offset:448
	buffer_store_dwordx2 v[30:31], v32, s[0:3], 0 offen offset:480
	s_endpgm
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel matmul_kernel
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
		.amdhsa_next_free_vgpr 146
		.amdhsa_next_free_sgpr 18
		.amdhsa_accum_offset 148
		.amdhsa_reserve_vcc 0
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
	.size	matmul_kernel, .Lfunc_end0-matmul_kernel
	.cfi_endproc
                                        ; -- End function
	.set matmul_kernel.num_vgpr, 146
	.set matmul_kernel.num_agpr, 0
	.set matmul_kernel.numbered_sgpr, 18
	.set matmul_kernel.num_named_barrier, 0
	.set matmul_kernel.private_seg_size, 0
	.set matmul_kernel.uses_vcc, 0
	.set matmul_kernel.uses_flat_scratch, 0
	.set matmul_kernel.has_dyn_sized_stack, 0
	.set matmul_kernel.has_recursion, 0
	.set matmul_kernel.has_indirect_call, 0
	.section	.AMDGPU.csdata,"",@progbits
; Kernel info:
; codeLenInByte = 2536
; TotalNumSgprs: 24
; NumVgprs: 146
; NumAgprs: 0
; TotalNumVgprs: 146
; ScratchSize: 0
; MemoryBound: 0
; FloatMode: 240
; IeeeMode: 1
; LDSByteSize: 0 bytes/workgroup (compile time only)
; SGPRBlocks: 2
; VGPRBlocks: 18
; NumSGPRsForWavesPerEU: 24
; NumVGPRsForWavesPerEU: 146
; AccumOffset: 148
; Occupancy: 3
; WaveLimiterHint : 0
; COMPUTE_PGM_RSRC2:SCRATCH_EN: 0
; COMPUTE_PGM_RSRC2:USER_SGPR: 16
; COMPUTE_PGM_RSRC2:TRAP_HANDLER: 0
; COMPUTE_PGM_RSRC2:TGID_X_EN: 1
; COMPUTE_PGM_RSRC2:TGID_Y_EN: 0
; COMPUTE_PGM_RSRC2:TGID_Z_EN: 0
; COMPUTE_PGM_RSRC2:TIDIG_COMP_CNT: 0
; COMPUTE_PGM_RSRC3_GFX90A:ACCUM_OFFSET: 36
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
	.section	.debug_info,"",@progbits
.Lcu_begin0:
	.long	.Ldebug_info_end0-.Ldebug_info_start0 ; Length of Unit
.Ldebug_info_start0:
	.short	4                               ; DWARF version number
	.long	.debug_abbrev                   ; Offset Into Abbrev. Section
	.byte	8                               ; Address Size (in bytes)
	.byte	1                               ; Abbrev [1] 0xb:0x75 DW_TAG_compile_unit
	.long	.Linfo_string0                  ; DW_AT_producer
	.short	2                               ; DW_AT_language
	.long	.Linfo_string1                  ; DW_AT_name
	.long	.Lline_table_start0             ; DW_AT_stmt_list
	.long	.Linfo_string2                  ; DW_AT_comp_dir
	.quad	.Lfunc_begin0                   ; DW_AT_low_pc
	.long	.Lfunc_end0-.Lfunc_begin0       ; DW_AT_high_pc
	.byte	2                               ; Abbrev [2] 0x2a:0x6 DW_TAG_subprogram
	.long	.Linfo_string3                  ; DW_AT_name
	.byte	1                               ; DW_AT_inline
	.byte	3                               ; Abbrev [3] 0x30:0x4f DW_TAG_subprogram
	.quad	.Lfunc_begin0                   ; DW_AT_low_pc
	.long	.Lfunc_end0-.Lfunc_begin0       ; DW_AT_high_pc
	.long	42                              ; DW_AT_abstract_origin
	.byte	4                               ; Abbrev [4] 0x41:0x3d DW_TAG_inlined_subroutine
	.long	42                              ; DW_AT_abstract_origin
	.quad	.Ltmp1                          ; DW_AT_low_pc
	.long	.Ltmp5-.Ltmp1                   ; DW_AT_high_pc
	.byte	1                               ; DW_AT_call_file
	.byte	48                              ; DW_AT_call_line
	.byte	71                              ; DW_AT_call_column
	.byte	5                               ; Abbrev [5] 0x55:0x14 DW_TAG_inlined_subroutine
	.long	42                              ; DW_AT_abstract_origin
	.quad	.Ltmp2                          ; DW_AT_low_pc
	.long	.Ltmp3-.Ltmp2                   ; DW_AT_high_pc
	.byte	1                               ; DW_AT_call_file
	.byte	15                              ; DW_AT_call_line
	.byte	27                              ; DW_AT_call_column
	.byte	5                               ; Abbrev [5] 0x69:0x14 DW_TAG_inlined_subroutine
	.long	42                              ; DW_AT_abstract_origin
	.quad	.Ltmp3                          ; DW_AT_low_pc
	.long	.Ltmp4-.Ltmp3                   ; DW_AT_high_pc
	.byte	1                               ; DW_AT_call_file
	.byte	14                              ; DW_AT_call_line
	.byte	27                              ; DW_AT_call_column
	.byte	0                               ; End Of Children Mark
	.byte	0                               ; End Of Children Mark
	.byte	0                               ; End Of Children Mark
.Ldebug_info_end0:
	.section	.debug_str,"MS",@progbits,1
.Linfo_string0:
	.asciz	"triton"                        ; string offset=0
.Linfo_string1:
	.asciz	"play_lds.py"                   ; string offset=7
.Linfo_string2:
	.asciz	"/root/OAI-triton/study_lds"    ; string offset=19
.Linfo_string3:
	.asciz	"matmul_kernel"                 ; string offset=46
	.section	".note.GNU-stack","",@progbits
	.amdgpu_metadata
---
amdhsa.kernels:
  - .agpr_count:     0
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
    .max_flat_workgroup_size: 64
    .name:           matmul_kernel
    .private_segment_fixed_size: 0
    .sgpr_count:     24
    .sgpr_spill_count: 0
    .symbol:         matmul_kernel.kd
    .uniform_work_group_size: 1
    .uses_dynamic_stack: false
    .vgpr_count:     146
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
