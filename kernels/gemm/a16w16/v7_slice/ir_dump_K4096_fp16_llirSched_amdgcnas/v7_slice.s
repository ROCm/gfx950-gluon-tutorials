	.amdgcn_target "amdgcn-amd-amdhsa--gfx950"
	.amdhsa_code_object_version 5
	.text
	.globl	v7_slice                        ; -- Begin function v7_slice
	.p2align	8
	.type	v7_slice,@function
v7_slice:                               ; @v7_slice
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
v_mov_b32_e32 v18, v0
s_nop 0
v_readfirstlane_b32 s22, v18
s_bfe_u32 s24, s22, 0x20006
s_add_i32 s0, s9, 0xff
s_ashr_i32 s1, s0, 31
s_lshr_b32 s1, s1, 24
s_add_i32 s0, s0, s1
s_ashr_i32 s0, s0, 8
s_xor_b32 s1, s16, s0
s_ashr_i32 s1, s1, 31
s_abs_i32 s8, s16
s_abs_i32 s9, s0
v_cvt_f32_u32_e32 v0, s9
v_rcp_iflag_f32_e32 v0, v0
s_nop 0
v_mul_f32_e32 v0, 0x4f7ffffe, v0
v_cvt_u32_f32_e32 v0, v0
s_mov_b32 s23, 0
s_sub_i32 s14, 0, s9
v_readfirstlane_b32 s15, v0
s_mul_i32 s14, s14, s15
s_mul_hi_u32 s14, s15, s14
s_add_i32 s15, s15, s14
s_mul_hi_u32 s14, s8, s15
s_mul_i32 s15, s14, s9
s_sub_i32 s8, s8, s15
s_add_i32 s15, s14, 1
s_sub_i32 s17, s8, s9
s_cmp_ge_u32 s8, s9
s_cselect_b32 s14, s15, s14
s_cselect_b32 s8, s17, s8
s_add_i32 s15, s14, 1
s_cmp_ge_u32 s8, s9
s_cselect_b32 s8, s15, s14
s_xor_b32 s8, s8, s1
s_sub_i32 s1, s8, s1
s_mul_i32 s0, s1, s0
s_sub_i32 s9, s16, s0
v_and_b32_e32 v0, 63, v18
v_lshl_or_b32 v19, s24, 6, v0
v_lshlrev_b32_e32 v0, 1, v18
v_and_b32_e32 v0, 0x70, v0
v_or_b32_e32 v1, s24, v0
v_or_b32_e32 v4, 4, v1
v_or_b32_e32 v6, 8, v1
v_or_b32_e32 v8, 12, v1
v_or_b32_e32 v2, 0x80, v1
v_or_b32_e32 v3, 0x84, v1
v_or_b32_e32 v5, 0x88, v1
v_or_b32_e32 v7, 0x8c, v1
v_lshlrev_b32_e32 v20, 3, v18
v_and_b32_e32 v0, 56, v20
s_lshl_b32 s15, s1, 8
s_mul_i32 s0, s15, s11
s_ashr_i32 s1, s0, 31
s_lshl_b64 s[0:1], s[0:1], 1
s_add_u32 s8, s2, s0
s_addc_u32 s41, s3, s1
s_lshl_b32 s14, s9, 8
s_mul_i32 s16, s14, s12
s_ashr_i32 s17, s16, 31
s_lshl_b64 s[20:21], s[16:17], 1
s_add_u32 s16, s4, s20
s_addc_u32 s42, s5, s21
v_mul_lo_u32 v10, v1, s11
v_mul_lo_u32 v11, v4, s11
v_mul_lo_u32 v12, v6, s11
v_mul_lo_u32 v13, v8, s11
v_mul_lo_u32 v14, v2, s11
v_mul_lo_u32 v15, v3, s11
v_mul_lo_u32 v16, v5, s11
v_mul_lo_u32 v17, v7, s11
v_mad_u64_u32 v[2:3], s[18:19], v1, s12, v[0:1]
v_mad_u64_u32 v[4:5], s[18:19], v4, s12, v[0:1]
v_mad_u64_u32 v[6:7], s[18:19], v6, s12, v[0:1]
v_mad_u64_u32 v[8:9], s[18:19], v8, s12, v[0:1]
s_lshl_b32 s9, s12, 8
s_ashr_i32 s12, s9, 1
s_add_i32 s57, s10, 63
s_and_b32 s9, s41, 0xffff
s_mov_b32 s11, 0x27000
s_mov_b32 s10, 0x7ffffffe
s_mul_i32 s53, s24, 0x420
s_add_i32 s24, s53, 0
v_add_lshl_u32 v254, v10, v0, 1
s_mov_b32 m0, s24
s_nop 0
buffer_load_dwordx4 v254, s[8:11], 0, offen, lds
s_add_i32 s54, s53, 0x1080
s_add_i32 s25, s24, 0x1080
v_add_lshl_u32 v26, v11, v0, 1
s_mov_b32 m0, s25
s_nop 0
buffer_load_dwordx4 v26, s[8:11], 0, offen, lds
s_add_i32 s55, s53, 0x2100
s_add_i32 s26, s24, 0x2100
v_add_lshl_u32 v27, v12, v0, 1
s_mov_b32 m0, s26
s_nop 0
buffer_load_dwordx4 v27, s[8:11], 0, offen, lds
s_add_i32 s56, s53, 0x3180
s_add_i32 s27, s24, 0x3180
v_add_lshl_u32 v29, v13, v0, 1
s_mov_b32 m0, s27
s_nop 0
buffer_load_dwordx4 v29, s[8:11], 0, offen, lds
s_add_i32 s28, s24, 0x4200
v_add_lshl_u32 v3, v14, v0, 1
s_mov_b32 m0, s28
s_nop 0
buffer_load_dwordx4 v3, s[8:11], 0, offen, lds
s_add_i32 s29, s24, 0x5280
v_add_lshl_u32 v5, v15, v0, 1
s_mov_b32 m0, s29
s_nop 0
buffer_load_dwordx4 v5, s[8:11], 0, offen, lds
s_add_i32 s30, s24, 0x6300
v_add_lshl_u32 v7, v16, v0, 1
s_mov_b32 m0, s30
s_nop 0
buffer_load_dwordx4 v7, s[8:11], 0, offen, lds
s_add_i32 s31, s24, 0x7380
v_add_lshl_u32 v9, v17, v0, 1
s_mov_b32 m0, s31
s_nop 0
buffer_load_dwordx4 v9, s[8:11], 0, offen, lds
s_and_b32 s17, s42, 0xffff
s_mov_b32 s18, s10
s_mov_b32 s19, s11
s_add_i32 s58, 0, 0x107e0
s_add_i32 s33, s58, s53
v_lshlrev_b32_e32 v10, 1, v2
s_mov_b32 m0, s33
s_nop 0
buffer_load_dwordx4 v10, s[16:19], 0, offen, lds
s_add_i32 s34, s58, s54
v_lshlrev_b32_e32 v11, 1, v4
s_mov_b32 m0, s34
s_nop 0
buffer_load_dwordx4 v11, s[16:19], 0, offen, lds
s_add_i32 s35, s58, s55
v_lshlrev_b32_e32 v12, 1, v6
s_mov_b32 m0, s35
s_nop 0
buffer_load_dwordx4 v12, s[16:19], 0, offen, lds
s_add_i32 s36, s58, s56
v_lshlrev_b32_e32 v13, 1, v8
s_mov_b32 m0, s36
s_nop 0
buffer_load_dwordx4 v13, s[16:19], 0, offen, lds
s_add_i32 s40, 0, 0x18bc0
s_add_i32 s37, s40, s53
v_add_lshl_u32 v14, v2, s12, 1
s_mov_b32 m0, s37
s_nop 0
buffer_load_dwordx4 v14, s[16:19], 0, offen, lds
s_add_i32 s38, s40, s54
v_add_lshl_u32 v4, v4, s12, 1
s_mov_b32 m0, s38
s_nop 0
buffer_load_dwordx4 v4, s[16:19], 0, offen, lds
s_add_i32 s39, s40, s55
v_add_lshl_u32 v6, v6, s12, 1
s_mov_b32 m0, s39
s_nop 0
buffer_load_dwordx4 v6, s[16:19], 0, offen, lds
s_add_i32 s40, s40, s56
v_add_lshl_u32 v8, v8, s12, 1
s_mov_b32 m0, s40
s_nop 0
buffer_load_dwordx4 v8, s[16:19], 0, offen, lds
s_add_u32 s8, s8, 0x80
s_addc_u32 s9, s41, 0
s_add_u32 s16, s16, 0x80
s_addc_u32 s12, s42, 0
s_waitcnt lgkmcnt(0)
s_barrier
s_and_b32 s9, s9, 0xffff
s_add_i32 s41, s37, 0xfffef840
s_mov_b32 m0, s41
s_nop 0
buffer_load_dwordx4 v254, s[8:11], 0, offen, lds
s_add_i32 s42, s37, 0xffff08c0
s_mov_b32 m0, s42
s_nop 0
buffer_load_dwordx4 v26, s[8:11], 0, offen, lds
s_add_i32 s43, s37, 0xffff1940
s_mov_b32 m0, s43
s_nop 0
buffer_load_dwordx4 v27, s[8:11], 0, offen, lds
s_add_i32 s44, s37, 0xffff29c0
s_mov_b32 m0, s44
s_nop 0
buffer_load_dwordx4 v29, s[8:11], 0, offen, lds
s_add_i32 s45, s37, 0xffff3a40
s_mov_b32 m0, s45
s_nop 0
buffer_load_dwordx4 v3, s[8:11], 0, offen, lds
s_add_i32 s46, s37, 0xffff4ac0
s_mov_b32 m0, s46
s_nop 0
buffer_load_dwordx4 v5, s[8:11], 0, offen, lds
s_add_i32 s47, s37, 0xffff5b40
s_mov_b32 m0, s47
s_nop 0
buffer_load_dwordx4 v7, s[8:11], 0, offen, lds
s_add_i32 s48, s37, 0xffff6bc0
s_mov_b32 m0, s48
s_nop 0
buffer_load_dwordx4 v9, s[8:11], 0, offen, lds
s_and_b32 s17, s12, 0xffff
s_add_i32 s52, 0, 0x149e0
s_add_i32 s49, s52, s53
s_mov_b32 m0, s49
s_nop 0
buffer_load_dwordx4 v10, s[16:19], 0, offen, lds
s_add_i32 s50, s52, s54
s_mov_b32 m0, s50
s_nop 0
buffer_load_dwordx4 v11, s[16:19], 0, offen, lds
s_add_i32 s51, s52, s55
s_mov_b32 m0, s51
s_nop 0
buffer_load_dwordx4 v12, s[16:19], 0, offen, lds
s_add_i32 s52, s52, s56
s_mov_b32 m0, s52
s_nop 0
buffer_load_dwordx4 v13, s[16:19], 0, offen, lds
s_add_i32 s8, 0, 0x1cdc0
s_add_i32 s53, s8, s53
s_mov_b32 m0, s53
s_nop 0
buffer_load_dwordx4 v14, s[16:19], 0, offen, lds
s_add_i32 s54, s8, s54
s_mov_b32 m0, s54
s_nop 0
buffer_load_dwordx4 v4, s[16:19], 0, offen, lds
s_add_i32 s55, s8, s55
s_mov_b32 m0, s55
s_nop 0
buffer_load_dwordx4 v6, s[16:19], 0, offen, lds
s_add_i32 s56, s8, s56
s_mov_b32 m0, s56
s_nop 0
buffer_load_dwordx4 v8, s[16:19], 0, offen, lds
s_waitcnt vmcnt(20), lgkmcnt(0)
s_barrier
v_and_b32_e32 v21, 15, v18
v_lshlrev_b32_e32 v1, 10, v21
s_movk_i32 s8, 0xb0
v_and_or_b32 v0, v19, s8, v1
v_lshlrev_b32_e32 v2, 5, v21
v_add_u32_e32 v0, v0, v2
v_add_u32_e32 v0, 0, v0
ds_read_b128 a[56:59], v0
ds_read_b128 a[60:63], v0, offset:64
ds_read_b128 a[48:51], v0, offset:256
ds_read_b128 a[52:55], v0, offset:320
ds_read_b128 a[40:43], v0, offset:512
ds_read_b128 a[44:47], v0, offset:576
ds_read_b128 a[32:35], v0, offset:768
ds_read_b128 a[36:39], v0, offset:832
ds_read_b128 a[24:27], v0, offset:16896
ds_read_b128 a[28:31], v0, offset:16960
ds_read_b128 a[16:19], v0, offset:17152
ds_read_b128 a[20:23], v0, offset:17216
ds_read_b128 a[8:11], v0, offset:17408
ds_read_b128 a[12:15], v0, offset:17472
ds_read_b128 a[0:3], v0, offset:17664
ds_read_b128 a[4:7], v0, offset:17728
s_and_b32 s12, s22, 64
v_and_or_b32 v1, v18, 48, v1
v_add_u32_e32 v1, v1, v2
v_lshl_add_u32 v1, s12, 1, v1
v_add_u32_e32 v2, s58, v1
ds_read_b128 a[88:91], v2
ds_read_b128 a[92:95], v2, offset:64
ds_read_b128 a[80:83], v2, offset:256
ds_read_b128 a[84:87], v2, offset:320
ds_read_b128 a[72:75], v2, offset:512
ds_read_b128 a[76:79], v2, offset:576
ds_read_b128 a[64:67], v2, offset:768
ds_read_b128 a[68:71], v2, offset:832
s_cmpk_lt_i32 s57, 0x80
v_add_u32_e32 v17, 0, v1
s_cbranch_scc1 .LBB0_4
; %bb.1:
v_accvgpr_write_b32 a156, v21
v_accvgpr_write_b32 a154, v20
v_accvgpr_write_b32 a153, v19
v_accvgpr_write_b32 a152, v18
s_ashr_i32 s8, s57, 31
s_lshr_b32 s8, s8, 26
s_add_i32 s57, s57, s8
s_ashr_i32 s17, s57, 6
s_add_i32 s16, s17, -1
s_add_i32 s17, s17, -2
s_add_u32 s4, s4, s20
s_addc_u32 s5, s5, s21
s_add_u32 s4, s4, 0x180
s_addc_u32 s5, s5, 0
s_add_u32 s0, s2, s0
s_addc_u32 s1, s3, s1
s_add_u32 s18, s0, 0x180
s_addc_u32 s19, s1, 0
v_mov_b32_e32 v126, 0
v_add_u32_e32 v1, 0x18bc0, v17
v_accvgpr_write_b32 a170, v1
v_add_u32_e32 v1, 0x149e0, v17
v_accvgpr_write_b32 a171, v1
v_add_u32_e32 v1, 0x1cdc0, v17
v_accvgpr_write_b32 a172, v1
v_accvgpr_write_b32 a155, v17
v_add_u32_e32 v1, 0x107e0, v17
v_accvgpr_write_b32 a173, v1
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
v_mov_b32_e32 v106, v126
v_mov_b32_e32 v107, v126
v_mov_b32_e32 v108, v126
v_mov_b32_e32 v109, v126
v_mov_b32_e32 v102, v126
v_mov_b32_e32 v103, v126
v_mov_b32_e32 v104, v126
v_mov_b32_e32 v105, v126
v_mov_b32_e32 v98, v126
v_mov_b32_e32 v99, v126
v_mov_b32_e32 v100, v126
v_mov_b32_e32 v101, v126
v_mov_b32_e32 v94, v126
v_mov_b32_e32 v95, v126
v_mov_b32_e32 v96, v126
v_mov_b32_e32 v97, v126
v_mov_b32_e32 v90, v126
v_mov_b32_e32 v91, v126
v_mov_b32_e32 v92, v126
v_mov_b32_e32 v93, v126
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
v_mov_b32_e32 v54, v126
v_mov_b32_e32 v55, v126
v_mov_b32_e32 v56, v126
v_mov_b32_e32 v57, v126
v_mov_b32_e32 v50, v126
v_mov_b32_e32 v51, v126
v_mov_b32_e32 v52, v126
v_mov_b32_e32 v53, v126
v_mov_b32_e32 v2, v126
v_accvgpr_write_b32 a174, v3
v_mov_b32_e32 v3, v126
v_accvgpr_write_b32 a175, v4
v_mov_b32_e32 v4, v126
v_accvgpr_write_b32 a176, v5
v_mov_b32_e32 v5, v126
v_accvgpr_write_b32 a127, v5
v_accvgpr_write_b32 a126, v4
v_accvgpr_write_b32 a125, v3
v_accvgpr_write_b32 a124, v2
v_accvgpr_write_b32 a169, v5
v_accvgpr_write_b32 a168, v4
v_accvgpr_write_b32 a167, v3
v_accvgpr_write_b32 a166, v2
v_accvgpr_write_b32 a165, v5
v_accvgpr_write_b32 a164, v4
v_accvgpr_write_b32 a163, v3
v_accvgpr_write_b32 a162, v2
v_accvgpr_write_b32 a161, v5
v_accvgpr_write_b32 a160, v4
v_accvgpr_write_b32 a159, v3
v_accvgpr_write_b32 a158, v2
v_accvgpr_write_b32 a132, v126
v_accvgpr_write_b32 a133, v126
v_accvgpr_write_b32 a134, v126
v_accvgpr_write_b32 a135, v126
v_accvgpr_write_b32 a177, v10
v_accvgpr_write_b32 a128, v126
v_accvgpr_write_b32 a178, v11
v_accvgpr_write_b32 a129, v126
v_accvgpr_write_b32 a179, v12
v_accvgpr_write_b32 a130, v126
v_accvgpr_write_b32 a180, v13
v_accvgpr_write_b32 a131, v126
v_accvgpr_write_b32 a96, v126
v_accvgpr_write_b32 a97, v126
v_accvgpr_write_b32 a98, v126
v_accvgpr_write_b32 a99, v126
v_mov_b32_e32 v16, v126
v_mov_b32_e32 v17, v126
v_mov_b32_e32 v18, v126
v_mov_b32_e32 v19, v126
v_accvgpr_write_b32 a189, v19
v_accvgpr_write_b32 a188, v18
v_accvgpr_write_b32 a187, v17
v_accvgpr_write_b32 a186, v16
v_mov_b32_e32 v250, v126
v_mov_b32_e32 v251, v126
v_mov_b32_e32 v252, v126
v_mov_b32_e32 v253, v126
v_mov_b32_e32 v246, v126
v_mov_b32_e32 v247, v126
v_mov_b32_e32 v248, v126
v_mov_b32_e32 v249, v126
v_mov_b32_e32 v242, v126
v_mov_b32_e32 v243, v126
v_mov_b32_e32 v244, v126
v_mov_b32_e32 v245, v126
v_accvgpr_write_b32 a210, v126
v_accvgpr_write_b32 a211, v126
v_accvgpr_write_b32 a212, v126
v_accvgpr_write_b32 a213, v126
v_accvgpr_write_b32 a185, v19
v_accvgpr_write_b32 a184, v18
v_accvgpr_write_b32 a183, v17
v_accvgpr_write_b32 a182, v16
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
v_accvgpr_write_b32 a226, v126
v_accvgpr_write_b32 a227, v126
v_accvgpr_write_b32 a228, v126
v_accvgpr_write_b32 a229, v126
v_accvgpr_write_b32 a181, v7
v_accvgpr_write_b32 a190, v9
v_accvgpr_write_b32 a192, v14
v_accvgpr_write_b32 a193, v6
v_accvgpr_write_b32 a191, v8
v_accvgpr_write_b32 a197, v126
v_accvgpr_write_b32 a195, v126
v_accvgpr_write_b32 a196, v126
v_accvgpr_write_b32 a194, v126
v_accvgpr_write_b32 a201, v126
v_accvgpr_write_b32 a200, v126
v_accvgpr_write_b32 a199, v126
v_accvgpr_write_b32 a198, v126
v_accvgpr_write_b32 a204, v126
v_accvgpr_write_b32 a202, v126
v_accvgpr_write_b32 a203, v126
v_accvgpr_write_b32 a205, v126
v_accvgpr_write_b32 a209, v126
v_accvgpr_write_b32 a206, v126
v_accvgpr_write_b32 a208, v126
v_accvgpr_write_b32 a207, v126
v_accvgpr_write_b32 a214, v126
v_accvgpr_write_b32 a216, v126
v_accvgpr_write_b32 a215, v126
v_accvgpr_write_b32 a217, v126
v_accvgpr_read_b32 v24, a170
v_bfrev_b32_e32 v3, 1
v_accvgpr_read_b32 v37, a174
v_accvgpr_read_b32 v4, a176
v_accvgpr_read_b32 v23, a181
v_accvgpr_read_b32 v36, a190
s_mov_b32 s3, s11
v_accvgpr_read_b32 v35, a177
v_accvgpr_read_b32 v39, a178
v_accvgpr_read_b32 v5, a179
v_accvgpr_read_b32 v34, a180
v_accvgpr_read_b32 v40, a171
v_accvgpr_read_b32 v31, a192
v_accvgpr_read_b32 v38, a175
v_accvgpr_read_b32 v32, a193
v_accvgpr_read_b32 v2, a191
v_accvgpr_read_b32 v33, a172
v_accvgpr_read_b32 v25, a173
.LBB0_2:
s_add_u32 s0, s4, 0xffffff80
s_addc_u32 s1, s5, -1
s_add_u32 s8, s18, 0xffffff80
s_addc_u32 s2, s19, -1
s_waitcnt lgkmcnt(0)
v_mfma_f32_16x16x32_f16 a[186:189], a[88:91], a[56:59], a[186:189]
v_mfma_f32_16x16x32_f16 a[186:189], a[92:95], a[60:63], a[186:189]
v_mfma_f32_16x16x32_f16 v[250:253], a[80:83], a[56:59], v[250:253]
v_mfma_f32_16x16x32_f16 v[250:253], a[84:87], a[60:63], v[250:253]
s_waitcnt vmcnt(16), lgkmcnt(0)
s_barrier
ds_read_b128 a[100:103], v24
v_mfma_f32_16x16x32_f16 v[246:249], a[72:75], a[56:59], v[246:249]
ds_read_b128 a[104:107], v24, offset:64
v_mfma_f32_16x16x32_f16 v[246:249], a[76:79], a[60:63], v[246:249]
ds_read_b128 a[108:111], v24, offset:256
v_mfma_f32_16x16x32_f16 v[242:245], a[64:67], a[56:59], v[242:245]
ds_read_b128 a[112:115], v24, offset:320
v_mfma_f32_16x16x32_f16 v[242:245], a[68:71], a[60:63], v[242:245]
ds_read_b128 a[116:119], v24, offset:512
v_mfma_f32_16x16x32_f16 a[210:213], a[88:91], a[48:51], a[210:213]
ds_read_b128 a[120:123], v24, offset:576
v_mfma_f32_16x16x32_f16 a[210:213], a[92:95], a[52:55], a[210:213]
ds_read_b128 a[136:139], v24, offset:768
v_mfma_f32_16x16x32_f16 a[182:185], a[80:83], a[48:51], a[182:185]
ds_read_b128 a[140:143], v24, offset:832
v_mfma_f32_16x16x32_f16 a[182:185], a[84:87], a[52:55], a[182:185]
v_mfma_f32_16x16x32_f16 v[230:233], a[72:75], a[48:51], v[230:233]
v_mfma_f32_16x16x32_f16 v[230:233], a[76:79], a[52:55], v[230:233]
s_cmp_eq_u32 s17, s23
s_cselect_b64 vcc, -1, 0
s_and_b32 s9, s2, 0xffff
v_cndmask_b32_e32 v28, v254, v3, vcc
s_mov_b32 m0, s24
v_mfma_f32_16x16x32_f16 v[226:229], a[64:67], a[48:51], v[226:229]
buffer_load_dwordx4 v254, s[8:11], 0, offen, lds
v_mfma_f32_16x16x32_f16 v[226:229], a[68:71], a[52:55], v[226:229]
v_mfma_f32_16x16x32_f16 v[222:225], a[88:91], a[40:43], v[222:225]
v_mfma_f32_16x16x32_f16 v[222:225], a[92:95], a[44:47], v[222:225]
v_cndmask_b32_e32 v26, v26, v3, vcc
s_mov_b32 m0, s25
v_mfma_f32_16x16x32_f16 v[218:221], a[80:83], a[40:43], v[218:221]
buffer_load_dwordx4 v26, s[8:11], 0, offen, lds
v_mfma_f32_16x16x32_f16 v[218:221], a[84:87], a[44:47], v[218:221]
v_mfma_f32_16x16x32_f16 v[214:217], a[72:75], a[40:43], v[214:217]
v_mfma_f32_16x16x32_f16 v[214:217], a[76:79], a[44:47], v[214:217]
v_cndmask_b32_e32 v27, v27, v3, vcc
s_mov_b32 m0, s26
v_mfma_f32_16x16x32_f16 v[210:213], a[64:67], a[40:43], v[210:213]
buffer_load_dwordx4 v27, s[8:11], 0, offen, lds
v_mfma_f32_16x16x32_f16 v[210:213], a[68:71], a[44:47], v[210:213]
v_mfma_f32_16x16x32_f16 v[206:209], a[88:91], a[32:35], v[206:209]
v_mfma_f32_16x16x32_f16 v[206:209], a[92:95], a[36:39], v[206:209]
v_cndmask_b32_e32 v29, v29, v3, vcc
s_mov_b32 m0, s27
v_mfma_f32_16x16x32_f16 v[202:205], a[80:83], a[32:35], v[202:205]
buffer_load_dwordx4 v29, s[8:11], 0, offen, lds
v_mfma_f32_16x16x32_f16 v[202:205], a[84:87], a[36:39], v[202:205]
v_mfma_f32_16x16x32_f16 v[198:201], a[72:75], a[32:35], v[198:201]
v_mfma_f32_16x16x32_f16 v[198:201], a[76:79], a[36:39], v[198:201]
v_cndmask_b32_e32 v37, v37, v3, vcc
s_mov_b32 m0, s28
v_mfma_f32_16x16x32_f16 v[194:197], a[64:67], a[32:35], v[194:197]
buffer_load_dwordx4 v37, s[8:11], 0, offen, lds
v_mfma_f32_16x16x32_f16 v[194:197], a[68:71], a[36:39], v[194:197]
v_mfma_f32_16x16x32_f16 v[190:193], a[88:91], a[24:27], v[190:193]
v_mfma_f32_16x16x32_f16 v[190:193], a[92:95], a[28:31], v[190:193]
v_cndmask_b32_e32 v4, v4, v3, vcc
s_mov_b32 m0, s29
v_mfma_f32_16x16x32_f16 v[186:189], a[80:83], a[24:27], v[186:189]
buffer_load_dwordx4 v4, s[8:11], 0, offen, lds
v_mfma_f32_16x16x32_f16 v[186:189], a[84:87], a[28:31], v[186:189]
v_mfma_f32_16x16x32_f16 v[182:185], a[72:75], a[24:27], v[182:185]
v_mfma_f32_16x16x32_f16 v[182:185], a[76:79], a[28:31], v[182:185]
v_cndmask_b32_e32 v23, v23, v3, vcc
s_mov_b32 m0, s30
v_mfma_f32_16x16x32_f16 v[178:181], a[64:67], a[24:27], v[178:181]
buffer_load_dwordx4 v23, s[8:11], 0, offen, lds
v_mfma_f32_16x16x32_f16 v[178:181], a[68:71], a[28:31], v[178:181]
v_mfma_f32_16x16x32_f16 v[174:177], a[88:91], a[16:19], v[174:177]
v_mfma_f32_16x16x32_f16 v[174:177], a[92:95], a[20:23], v[174:177]
v_cndmask_b32_e32 v36, v36, v3, vcc
s_mov_b32 m0, s31
v_mfma_f32_16x16x32_f16 v[170:173], a[80:83], a[16:19], v[170:173]
buffer_load_dwordx4 v36, s[8:11], 0, offen, lds
v_mfma_f32_16x16x32_f16 v[170:173], a[84:87], a[20:23], v[170:173]
v_mfma_f32_16x16x32_f16 v[166:169], a[72:75], a[16:19], v[166:169]
v_mfma_f32_16x16x32_f16 v[166:169], a[76:79], a[20:23], v[166:169]
s_and_b32 s1, s1, 0xffff
s_mov_b32 s2, s10
v_cndmask_b32_e32 v35, v35, v3, vcc
s_mov_b32 m0, s33
v_mfma_f32_16x16x32_f16 v[162:165], a[64:67], a[16:19], v[162:165]
buffer_load_dwordx4 v35, s[0:3], 0, offen, lds
v_mfma_f32_16x16x32_f16 v[162:165], a[68:71], a[20:23], v[162:165]
v_mfma_f32_16x16x32_f16 v[158:161], a[88:91], a[8:11], v[158:161]
v_mfma_f32_16x16x32_f16 v[158:161], a[92:95], a[12:15], v[158:161]
v_cndmask_b32_e32 v39, v39, v3, vcc
s_mov_b32 m0, s34
v_mfma_f32_16x16x32_f16 v[154:157], a[80:83], a[8:11], v[154:157]
buffer_load_dwordx4 v39, s[0:3], 0, offen, lds
v_mfma_f32_16x16x32_f16 v[154:157], a[84:87], a[12:15], v[154:157]
v_mfma_f32_16x16x32_f16 v[150:153], a[72:75], a[8:11], v[150:153]
v_mfma_f32_16x16x32_f16 v[150:153], a[76:79], a[12:15], v[150:153]
v_cndmask_b32_e32 v5, v5, v3, vcc
s_mov_b32 m0, s35
v_mfma_f32_16x16x32_f16 v[146:149], a[64:67], a[8:11], v[146:149]
buffer_load_dwordx4 v5, s[0:3], 0, offen, lds
v_mfma_f32_16x16x32_f16 v[146:149], a[68:71], a[12:15], v[146:149]
v_mfma_f32_16x16x32_f16 v[142:145], a[88:91], a[0:3], v[142:145]
v_mfma_f32_16x16x32_f16 v[142:145], a[92:95], a[4:7], v[142:145]
v_cndmask_b32_e32 v34, v34, v3, vcc
s_mov_b32 m0, s36
v_mfma_f32_16x16x32_f16 v[138:141], a[80:83], a[0:3], v[138:141]
buffer_load_dwordx4 v34, s[0:3], 0, offen, lds
v_mfma_f32_16x16x32_f16 v[138:141], a[84:87], a[4:7], v[138:141]
v_mfma_f32_16x16x32_f16 v[134:137], a[72:75], a[0:3], v[134:137]
v_mfma_f32_16x16x32_f16 v[134:137], a[76:79], a[4:7], v[134:137]
v_mfma_f32_16x16x32_f16 a[226:229], a[64:67], a[0:3], a[226:229]
v_mfma_f32_16x16x32_f16 a[226:229], a[68:71], a[4:7], a[226:229]
s_waitcnt lgkmcnt(0)
v_mfma_f32_16x16x32_f16 v[126:129], a[100:103], a[56:59], v[126:129]
v_mfma_f32_16x16x32_f16 v[126:129], a[104:107], a[60:63], v[126:129]
v_mfma_f32_16x16x32_f16 v[122:125], a[108:111], a[56:59], v[122:125]
v_mfma_f32_16x16x32_f16 v[122:125], a[112:115], a[60:63], v[122:125]
v_mfma_f32_16x16x32_f16 v[118:121], a[116:119], a[56:59], v[118:121]
v_mfma_f32_16x16x32_f16 v[118:121], a[120:123], a[60:63], v[118:121]
v_mfma_f32_16x16x32_f16 v[114:117], a[136:139], a[56:59], v[114:117]
v_mfma_f32_16x16x32_f16 v[114:117], a[140:143], a[60:63], v[114:117]
v_mfma_f32_16x16x32_f16 v[110:113], a[100:103], a[48:51], v[110:113]
v_mfma_f32_16x16x32_f16 v[110:113], a[104:107], a[52:55], v[110:113]
v_mfma_f32_16x16x32_f16 v[106:109], a[108:111], a[48:51], v[106:109]
v_mfma_f32_16x16x32_f16 v[106:109], a[112:115], a[52:55], v[106:109]
v_mfma_f32_16x16x32_f16 v[102:105], a[116:119], a[48:51], v[102:105]
v_mfma_f32_16x16x32_f16 v[102:105], a[120:123], a[52:55], v[102:105]
v_mfma_f32_16x16x32_f16 v[98:101], a[136:139], a[48:51], v[98:101]
v_mfma_f32_16x16x32_f16 v[98:101], a[140:143], a[52:55], v[98:101]
v_mfma_f32_16x16x32_f16 v[94:97], a[100:103], a[40:43], v[94:97]
v_mfma_f32_16x16x32_f16 v[94:97], a[104:107], a[44:47], v[94:97]
v_mfma_f32_16x16x32_f16 v[90:93], a[108:111], a[40:43], v[90:93]
v_mfma_f32_16x16x32_f16 v[90:93], a[112:115], a[44:47], v[90:93]
s_waitcnt vmcnt(16), lgkmcnt(0)
s_barrier
ds_read_b128 a[144:147], v0, offset:33792
v_mfma_f32_16x16x32_f16 v[86:89], a[116:119], a[40:43], v[86:89]
ds_read_b128 a[148:151], v0, offset:33856
v_mfma_f32_16x16x32_f16 v[86:89], a[120:123], a[44:47], v[86:89]
ds_read_b128 a[170:173], v0, offset:34048
v_mfma_f32_16x16x32_f16 v[82:85], a[136:139], a[40:43], v[82:85]
ds_read_b128 a[174:177], v0, offset:34112
v_mfma_f32_16x16x32_f16 v[82:85], a[140:143], a[44:47], v[82:85]
ds_read_b128 a[178:181], v0, offset:34304
v_mfma_f32_16x16x32_f16 v[78:81], a[100:103], a[32:35], v[78:81]
ds_read_b128 a[190:193], v0, offset:34368
v_mfma_f32_16x16x32_f16 v[78:81], a[104:107], a[36:39], v[78:81]
ds_read_b128 a[218:221], v0, offset:34560
v_mfma_f32_16x16x32_f16 v[74:77], a[108:111], a[32:35], v[74:77]
ds_read_b128 a[222:225], v0, offset:34624
v_mfma_f32_16x16x32_f16 v[74:77], a[112:115], a[36:39], v[74:77]
ds_read_b128 a[230:233], v0, offset:50688
v_mfma_f32_16x16x32_f16 v[70:73], a[116:119], a[32:35], v[70:73]
ds_read_b128 a[234:237], v0, offset:50752
v_mfma_f32_16x16x32_f16 v[70:73], a[120:123], a[36:39], v[70:73]
ds_read_b128 a[238:241], v0, offset:50944
v_mfma_f32_16x16x32_f16 v[66:69], a[136:139], a[32:35], v[66:69]
ds_read_b128 a[242:245], v0, offset:51008
v_mfma_f32_16x16x32_f16 v[66:69], a[140:143], a[36:39], v[66:69]
ds_read_b128 a[246:249], v0, offset:51200
v_mfma_f32_16x16x32_f16 v[62:65], a[100:103], a[24:27], v[62:65]
ds_read_b128 a[250:253], v0, offset:51264
v_mfma_f32_16x16x32_f16 v[62:65], a[104:107], a[28:31], v[62:65]
ds_read_b128 v[6:9], v0, offset:51456
v_mfma_f32_16x16x32_f16 v[58:61], a[108:111], a[24:27], v[58:61]
ds_read_b128 v[42:45], v0, offset:51520
v_mfma_f32_16x16x32_f16 v[58:61], a[112:115], a[28:31], v[58:61]
ds_read_b128 a[88:91], v40
v_mfma_f32_16x16x32_f16 v[54:57], a[116:119], a[24:27], v[54:57]
ds_read_b128 a[92:95], v40, offset:64
v_mfma_f32_16x16x32_f16 v[54:57], a[120:123], a[28:31], v[54:57]
ds_read_b128 a[80:83], v40, offset:256
v_mfma_f32_16x16x32_f16 v[50:53], a[136:139], a[24:27], v[50:53]
ds_read_b128 a[84:87], v40, offset:320
v_mfma_f32_16x16x32_f16 v[50:53], a[140:143], a[28:31], v[50:53]
ds_read_b128 a[72:75], v40, offset:512
v_mfma_f32_16x16x32_f16 a[214:217], a[100:103], a[16:19], a[214:217]
ds_read_b128 a[76:79], v40, offset:576
v_mfma_f32_16x16x32_f16 a[214:217], a[104:107], a[20:23], a[214:217]
ds_read_b128 a[64:67], v40, offset:768
v_mfma_f32_16x16x32_f16 a[206:209], a[108:111], a[16:19], a[206:209]
ds_read_b128 a[68:71], v40, offset:832
v_mfma_f32_16x16x32_f16 a[206:209], a[112:115], a[20:23], a[206:209]
v_mfma_f32_16x16x32_f16 a[202:205], a[116:119], a[16:19], a[202:205]
v_mfma_f32_16x16x32_f16 a[202:205], a[120:123], a[20:23], a[202:205]
v_cndmask_b32_e32 v31, v31, v3, vcc
s_mov_b32 m0, s37
v_mfma_f32_16x16x32_f16 a[198:201], a[136:139], a[16:19], a[198:201]
buffer_load_dwordx4 v31, s[0:3], 0, offen, lds
v_mfma_f32_16x16x32_f16 a[198:201], a[140:143], a[20:23], a[198:201]
v_mfma_f32_16x16x32_f16 a[194:197], a[100:103], a[8:11], a[194:197]
v_mfma_f32_16x16x32_f16 a[194:197], a[104:107], a[12:15], a[194:197]
v_cndmask_b32_e32 v38, v38, v3, vcc
s_mov_b32 m0, s38
v_mfma_f32_16x16x32_f16 a[124:127], a[108:111], a[8:11], a[124:127]
buffer_load_dwordx4 v38, s[0:3], 0, offen, lds
v_mfma_f32_16x16x32_f16 a[124:127], a[112:115], a[12:15], a[124:127]
v_mfma_f32_16x16x32_f16 a[166:169], a[116:119], a[8:11], a[166:169]
v_mfma_f32_16x16x32_f16 a[166:169], a[120:123], a[12:15], a[166:169]
v_cndmask_b32_e32 v32, v32, v3, vcc
s_mov_b32 m0, s39
v_mfma_f32_16x16x32_f16 a[162:165], a[136:139], a[8:11], a[162:165]
buffer_load_dwordx4 v32, s[0:3], 0, offen, lds
v_mfma_f32_16x16x32_f16 a[162:165], a[140:143], a[12:15], a[162:165]
v_mfma_f32_16x16x32_f16 a[158:161], a[100:103], a[0:3], a[158:161]
v_mfma_f32_16x16x32_f16 a[158:161], a[104:107], a[4:7], a[158:161]
v_cndmask_b32_e32 v2, v2, v3, vcc
s_mov_b32 m0, s40
v_mfma_f32_16x16x32_f16 a[132:135], a[108:111], a[0:3], a[132:135]
buffer_load_dwordx4 v2, s[0:3], 0, offen, lds
v_mfma_f32_16x16x32_f16 a[132:135], a[112:115], a[4:7], a[132:135]
v_mfma_f32_16x16x32_f16 a[128:131], a[116:119], a[0:3], a[128:131]
v_mfma_f32_16x16x32_f16 a[128:131], a[120:123], a[4:7], a[128:131]
v_mfma_f32_16x16x32_f16 a[96:99], a[136:139], a[0:3], a[96:99]
v_mfma_f32_16x16x32_f16 a[96:99], a[140:143], a[4:7], a[96:99]
s_waitcnt lgkmcnt(0)
v_mfma_f32_16x16x32_f16 a[186:189], a[88:91], a[144:147], a[186:189]
v_mfma_f32_16x16x32_f16 a[186:189], a[92:95], a[148:151], a[186:189]
v_mfma_f32_16x16x32_f16 v[250:253], a[80:83], a[144:147], v[250:253]
v_mfma_f32_16x16x32_f16 v[250:253], a[84:87], a[148:151], v[250:253]
s_waitcnt vmcnt(16), lgkmcnt(0)
s_barrier
ds_read_b128 a[100:103], v33
v_mfma_f32_16x16x32_f16 v[246:249], a[72:75], a[144:147], v[246:249]
ds_read_b128 a[104:107], v33, offset:64
v_mfma_f32_16x16x32_f16 v[246:249], a[76:79], a[148:151], v[246:249]
ds_read_b128 a[108:111], v33, offset:256
v_mfma_f32_16x16x32_f16 v[242:245], a[64:67], a[144:147], v[242:245]
ds_read_b128 a[112:115], v33, offset:320
v_mfma_f32_16x16x32_f16 v[242:245], a[68:71], a[148:151], v[242:245]
ds_read_b128 a[116:119], v33, offset:512
v_mfma_f32_16x16x32_f16 a[210:213], a[88:91], a[170:173], a[210:213]
ds_read_b128 a[120:123], v33, offset:576
v_mfma_f32_16x16x32_f16 a[210:213], a[92:95], a[174:177], a[210:213]
ds_read_b128 a[136:139], v33, offset:768
v_mfma_f32_16x16x32_f16 a[182:185], a[80:83], a[170:173], a[182:185]
ds_read_b128 a[140:143], v33, offset:832
v_mfma_f32_16x16x32_f16 a[182:185], a[84:87], a[174:177], a[182:185]
v_mfma_f32_16x16x32_f16 v[230:233], a[72:75], a[170:173], v[230:233]
v_mfma_f32_16x16x32_f16 v[230:233], a[76:79], a[174:177], v[230:233]
s_and_b32 s9, s19, 0xffff
s_mov_b32 s8, s18
s_mov_b32 m0, s41
v_mfma_f32_16x16x32_f16 v[226:229], a[64:67], a[170:173], v[226:229]
buffer_load_dwordx4 v254, s[8:11], 0, offen, lds
v_mfma_f32_16x16x32_f16 v[226:229], a[68:71], a[174:177], v[226:229]
v_mfma_f32_16x16x32_f16 v[222:225], a[88:91], a[178:181], v[222:225]
v_mfma_f32_16x16x32_f16 v[222:225], a[92:95], a[190:193], v[222:225]
s_mov_b32 m0, s42
v_mfma_f32_16x16x32_f16 v[218:221], a[80:83], a[178:181], v[218:221]
buffer_load_dwordx4 v26, s[8:11], 0, offen, lds
v_mfma_f32_16x16x32_f16 v[218:221], a[84:87], a[190:193], v[218:221]
v_mfma_f32_16x16x32_f16 v[214:217], a[72:75], a[178:181], v[214:217]
v_mfma_f32_16x16x32_f16 v[214:217], a[76:79], a[190:193], v[214:217]
s_mov_b32 m0, s43
v_mfma_f32_16x16x32_f16 v[210:213], a[64:67], a[178:181], v[210:213]
buffer_load_dwordx4 v27, s[8:11], 0, offen, lds
v_mfma_f32_16x16x32_f16 v[210:213], a[68:71], a[190:193], v[210:213]
v_mfma_f32_16x16x32_f16 v[206:209], a[88:91], a[218:221], v[206:209]
v_mfma_f32_16x16x32_f16 v[206:209], a[92:95], a[222:225], v[206:209]
s_mov_b32 m0, s44
v_mfma_f32_16x16x32_f16 v[202:205], a[80:83], a[218:221], v[202:205]
buffer_load_dwordx4 v29, s[8:11], 0, offen, lds
v_mfma_f32_16x16x32_f16 v[202:205], a[84:87], a[222:225], v[202:205]
v_mfma_f32_16x16x32_f16 v[198:201], a[72:75], a[218:221], v[198:201]
v_mfma_f32_16x16x32_f16 v[198:201], a[76:79], a[222:225], v[198:201]
s_mov_b32 m0, s45
v_mfma_f32_16x16x32_f16 v[194:197], a[64:67], a[218:221], v[194:197]
buffer_load_dwordx4 v37, s[8:11], 0, offen, lds
v_mfma_f32_16x16x32_f16 v[194:197], a[68:71], a[222:225], v[194:197]
v_mfma_f32_16x16x32_f16 v[190:193], a[88:91], a[230:233], v[190:193]
v_mfma_f32_16x16x32_f16 v[190:193], a[92:95], a[234:237], v[190:193]
s_mov_b32 m0, s46
v_mfma_f32_16x16x32_f16 v[186:189], a[80:83], a[230:233], v[186:189]
buffer_load_dwordx4 v4, s[8:11], 0, offen, lds
v_mfma_f32_16x16x32_f16 v[186:189], a[84:87], a[234:237], v[186:189]
v_mfma_f32_16x16x32_f16 v[182:185], a[72:75], a[230:233], v[182:185]
v_mfma_f32_16x16x32_f16 v[182:185], a[76:79], a[234:237], v[182:185]
s_mov_b32 m0, s47
v_mfma_f32_16x16x32_f16 v[178:181], a[64:67], a[230:233], v[178:181]
buffer_load_dwordx4 v23, s[8:11], 0, offen, lds
v_mfma_f32_16x16x32_f16 v[178:181], a[68:71], a[234:237], v[178:181]
v_mfma_f32_16x16x32_f16 v[174:177], a[88:91], a[238:241], v[174:177]
v_mfma_f32_16x16x32_f16 v[174:177], a[92:95], a[242:245], v[174:177]
s_mov_b32 m0, s48
v_mfma_f32_16x16x32_f16 v[170:173], a[80:83], a[238:241], v[170:173]
buffer_load_dwordx4 v36, s[8:11], 0, offen, lds
v_mfma_f32_16x16x32_f16 v[170:173], a[84:87], a[242:245], v[170:173]
v_mfma_f32_16x16x32_f16 v[166:169], a[72:75], a[238:241], v[166:169]
v_mfma_f32_16x16x32_f16 v[166:169], a[76:79], a[242:245], v[166:169]
s_and_b32 s9, s5, 0xffff
s_mov_b32 s8, s4
s_mov_b32 m0, s49
v_mfma_f32_16x16x32_f16 v[162:165], a[64:67], a[238:241], v[162:165]
buffer_load_dwordx4 v35, s[8:11], 0, offen, lds
v_mfma_f32_16x16x32_f16 v[162:165], a[68:71], a[242:245], v[162:165]
v_mfma_f32_16x16x32_f16 v[158:161], a[88:91], a[246:249], v[158:161]
v_mfma_f32_16x16x32_f16 v[158:161], a[92:95], a[250:253], v[158:161]
s_mov_b32 m0, s50
v_mfma_f32_16x16x32_f16 v[154:157], a[80:83], a[246:249], v[154:157]
buffer_load_dwordx4 v39, s[8:11], 0, offen, lds
v_mfma_f32_16x16x32_f16 v[154:157], a[84:87], a[250:253], v[154:157]
v_mfma_f32_16x16x32_f16 v[150:153], a[72:75], a[246:249], v[150:153]
v_mfma_f32_16x16x32_f16 v[150:153], a[76:79], a[250:253], v[150:153]
s_mov_b32 m0, s51
v_mfma_f32_16x16x32_f16 v[146:149], a[64:67], a[246:249], v[146:149]
buffer_load_dwordx4 v5, s[8:11], 0, offen, lds
v_mfma_f32_16x16x32_f16 v[146:149], a[68:71], a[250:253], v[146:149]
v_mfma_f32_16x16x32_f16 v[142:145], a[88:91], v[6:9], v[142:145]
v_mfma_f32_16x16x32_f16 v[142:145], a[92:95], v[42:45], v[142:145]
s_mov_b32 m0, s52
v_mfma_f32_16x16x32_f16 v[138:141], a[80:83], v[6:9], v[138:141]
buffer_load_dwordx4 v34, s[8:11], 0, offen, lds
v_mfma_f32_16x16x32_f16 v[138:141], a[84:87], v[42:45], v[138:141]
v_mfma_f32_16x16x32_f16 v[134:137], a[72:75], v[6:9], v[134:137]
v_mfma_f32_16x16x32_f16 v[134:137], a[76:79], v[42:45], v[134:137]
v_mfma_f32_16x16x32_f16 a[226:229], a[64:67], v[6:9], a[226:229]
v_mfma_f32_16x16x32_f16 a[226:229], a[68:71], v[42:45], a[226:229]
s_waitcnt lgkmcnt(0)
v_mfma_f32_16x16x32_f16 v[126:129], a[100:103], a[144:147], v[126:129]
v_mfma_f32_16x16x32_f16 v[126:129], a[104:107], a[148:151], v[126:129]
v_mfma_f32_16x16x32_f16 v[122:125], a[108:111], a[144:147], v[122:125]
v_mfma_f32_16x16x32_f16 v[122:125], a[112:115], a[148:151], v[122:125]
v_mfma_f32_16x16x32_f16 v[118:121], a[116:119], a[144:147], v[118:121]
v_mfma_f32_16x16x32_f16 v[118:121], a[120:123], a[148:151], v[118:121]
v_mfma_f32_16x16x32_f16 v[114:117], a[136:139], a[144:147], v[114:117]
v_mfma_f32_16x16x32_f16 v[114:117], a[140:143], a[148:151], v[114:117]
v_mfma_f32_16x16x32_f16 v[110:113], a[100:103], a[170:173], v[110:113]
v_mfma_f32_16x16x32_f16 v[110:113], a[104:107], a[174:177], v[110:113]
v_mfma_f32_16x16x32_f16 v[106:109], a[108:111], a[170:173], v[106:109]
v_mfma_f32_16x16x32_f16 v[106:109], a[112:115], a[174:177], v[106:109]
v_mfma_f32_16x16x32_f16 v[102:105], a[116:119], a[170:173], v[102:105]
v_mfma_f32_16x16x32_f16 v[102:105], a[120:123], a[174:177], v[102:105]
v_mfma_f32_16x16x32_f16 v[98:101], a[136:139], a[170:173], v[98:101]
v_mfma_f32_16x16x32_f16 v[98:101], a[140:143], a[174:177], v[98:101]
v_mfma_f32_16x16x32_f16 v[94:97], a[100:103], a[178:181], v[94:97]
v_mfma_f32_16x16x32_f16 v[94:97], a[104:107], a[190:193], v[94:97]
v_mfma_f32_16x16x32_f16 v[90:93], a[108:111], a[178:181], v[90:93]
v_mfma_f32_16x16x32_f16 v[90:93], a[112:115], a[190:193], v[90:93]
s_waitcnt vmcnt(16), lgkmcnt(0)
s_barrier
ds_read_b128 a[56:59], v0
v_mfma_f32_16x16x32_f16 v[86:89], a[116:119], a[178:181], v[86:89]
ds_read_b128 a[60:63], v0, offset:64
v_mfma_f32_16x16x32_f16 v[86:89], a[120:123], a[190:193], v[86:89]
ds_read_b128 a[48:51], v0, offset:256
v_mfma_f32_16x16x32_f16 v[82:85], a[136:139], a[178:181], v[82:85]
ds_read_b128 a[52:55], v0, offset:320
v_mfma_f32_16x16x32_f16 v[82:85], a[140:143], a[190:193], v[82:85]
ds_read_b128 a[40:43], v0, offset:512
v_mfma_f32_16x16x32_f16 v[78:81], a[100:103], a[218:221], v[78:81]
ds_read_b128 a[44:47], v0, offset:576
v_mfma_f32_16x16x32_f16 v[78:81], a[104:107], a[222:225], v[78:81]
ds_read_b128 a[32:35], v0, offset:768
v_mfma_f32_16x16x32_f16 v[74:77], a[108:111], a[218:221], v[74:77]
ds_read_b128 a[36:39], v0, offset:832
v_mfma_f32_16x16x32_f16 v[74:77], a[112:115], a[222:225], v[74:77]
ds_read_b128 a[24:27], v0, offset:16896
v_mfma_f32_16x16x32_f16 v[70:73], a[116:119], a[218:221], v[70:73]
ds_read_b128 a[28:31], v0, offset:16960
v_mfma_f32_16x16x32_f16 v[70:73], a[120:123], a[222:225], v[70:73]
ds_read_b128 a[16:19], v0, offset:17152
v_mfma_f32_16x16x32_f16 v[66:69], a[136:139], a[218:221], v[66:69]
ds_read_b128 a[20:23], v0, offset:17216
v_mfma_f32_16x16x32_f16 v[66:69], a[140:143], a[222:225], v[66:69]
ds_read_b128 a[8:11], v0, offset:17408
v_mfma_f32_16x16x32_f16 v[62:65], a[100:103], a[230:233], v[62:65]
ds_read_b128 a[12:15], v0, offset:17472
v_mfma_f32_16x16x32_f16 v[62:65], a[104:107], a[234:237], v[62:65]
ds_read_b128 a[0:3], v0, offset:17664
v_mfma_f32_16x16x32_f16 v[58:61], a[108:111], a[230:233], v[58:61]
ds_read_b128 a[4:7], v0, offset:17728
v_mfma_f32_16x16x32_f16 v[58:61], a[112:115], a[234:237], v[58:61]
ds_read_b128 a[88:91], v25
v_mfma_f32_16x16x32_f16 v[54:57], a[116:119], a[230:233], v[54:57]
ds_read_b128 a[92:95], v25, offset:64
v_mfma_f32_16x16x32_f16 v[54:57], a[120:123], a[234:237], v[54:57]
ds_read_b128 a[80:83], v25, offset:256
v_mfma_f32_16x16x32_f16 v[50:53], a[136:139], a[230:233], v[50:53]
ds_read_b128 a[84:87], v25, offset:320
v_mfma_f32_16x16x32_f16 v[50:53], a[140:143], a[234:237], v[50:53]
ds_read_b128 a[72:75], v25, offset:512
v_mfma_f32_16x16x32_f16 a[214:217], a[100:103], a[238:241], a[214:217]
ds_read_b128 a[76:79], v25, offset:576
v_mfma_f32_16x16x32_f16 a[214:217], a[104:107], a[242:245], a[214:217]
ds_read_b128 a[64:67], v25, offset:768
v_mfma_f32_16x16x32_f16 a[206:209], a[108:111], a[238:241], a[206:209]
ds_read_b128 a[68:71], v25, offset:832
v_mfma_f32_16x16x32_f16 a[206:209], a[112:115], a[242:245], a[206:209]
v_mfma_f32_16x16x32_f16 a[202:205], a[116:119], a[238:241], a[202:205]
v_mfma_f32_16x16x32_f16 a[202:205], a[120:123], a[242:245], a[202:205]
s_mov_b32 m0, s53
v_mfma_f32_16x16x32_f16 a[198:201], a[136:139], a[238:241], a[198:201]
buffer_load_dwordx4 v31, s[8:11], 0, offen, lds
v_mfma_f32_16x16x32_f16 a[198:201], a[140:143], a[242:245], a[198:201]
v_mfma_f32_16x16x32_f16 a[194:197], a[100:103], a[246:249], a[194:197]
v_mfma_f32_16x16x32_f16 a[194:197], a[104:107], a[250:253], a[194:197]
s_mov_b32 m0, s54
v_mfma_f32_16x16x32_f16 a[124:127], a[108:111], a[246:249], a[124:127]
buffer_load_dwordx4 v38, s[8:11], 0, offen, lds
v_mfma_f32_16x16x32_f16 a[124:127], a[112:115], a[250:253], a[124:127]
v_mfma_f32_16x16x32_f16 a[166:169], a[116:119], a[246:249], a[166:169]
v_mfma_f32_16x16x32_f16 a[166:169], a[120:123], a[250:253], a[166:169]
s_mov_b32 m0, s55
v_mfma_f32_16x16x32_f16 a[162:165], a[136:139], a[246:249], a[162:165]
buffer_load_dwordx4 v32, s[8:11], 0, offen, lds
v_mfma_f32_16x16x32_f16 a[162:165], a[140:143], a[250:253], a[162:165]
v_mfma_f32_16x16x32_f16 a[158:161], a[100:103], v[6:9], a[158:161]
v_mfma_f32_16x16x32_f16 a[158:161], a[104:107], v[42:45], a[158:161]
s_mov_b32 m0, s56
v_mfma_f32_16x16x32_f16 a[132:135], a[108:111], v[6:9], a[132:135]
buffer_load_dwordx4 v2, s[8:11], 0, offen, lds
v_mfma_f32_16x16x32_f16 a[132:135], a[112:115], v[42:45], a[132:135]
v_mfma_f32_16x16x32_f16 a[128:131], a[116:119], v[6:9], a[128:131]
v_mfma_f32_16x16x32_f16 a[128:131], a[120:123], v[42:45], a[128:131]
v_mfma_f32_16x16x32_f16 a[96:99], a[136:139], v[6:9], a[96:99]
v_mfma_f32_16x16x32_f16 a[96:99], a[140:143], v[42:45], a[96:99]
s_add_i32 s23, s23, 2
s_add_u32 s4, s4, 0x100
s_addc_u32 s5, s5, 0
s_add_u32 s18, s18, 0x100
s_addc_u32 s19, s19, 0
s_cmp_lt_i32 s23, s16
s_cbranch_scc1 .LBB0_2
; %bb.3:
v_accvgpr_read_b32 v46, a214
v_accvgpr_read_b32 v47, a215
v_accvgpr_read_b32 v48, a216
v_accvgpr_read_b32 v49, a217
v_accvgpr_read_b32 v42, a206
v_accvgpr_read_b32 v43, a207
v_accvgpr_read_b32 v44, a208
v_accvgpr_read_b32 v45, a209
v_accvgpr_read_b32 v38, a202
v_accvgpr_read_b32 v39, a203
v_accvgpr_read_b32 v40, a204
v_accvgpr_read_b32 v41, a205
v_accvgpr_read_b32 v34, a198
v_accvgpr_read_b32 v35, a199
v_accvgpr_read_b32 v36, a200
v_accvgpr_read_b32 v37, a201
v_accvgpr_read_b32 v30, a194
v_accvgpr_read_b32 v31, a195
v_accvgpr_read_b32 v32, a196
v_accvgpr_read_b32 v33, a197
v_accvgpr_mov_b32 a120, a210
v_accvgpr_mov_b32 a121, a211
v_accvgpr_mov_b32 a122, a212
v_accvgpr_mov_b32 a123, a213
v_accvgpr_read_b32 v18, a152
v_accvgpr_read_b32 v19, a153
v_accvgpr_read_b32 v20, a154
v_accvgpr_read_b32 v17, a155
v_accvgpr_read_b32 v21, a156
v_accvgpr_read_b32 v237, a185
v_accvgpr_read_b32 v236, a184
v_accvgpr_read_b32 v235, a183
v_accvgpr_read_b32 v234, a182
v_accvgpr_read_b32 v26, a186
v_accvgpr_read_b32 v27, a187
v_accvgpr_read_b32 v28, a188
v_accvgpr_read_b32 v29, a189
s_branch .LBB0_5
.LBB0_4:
v_accvgpr_write_b32 a99, 0
v_accvgpr_mov_b32 a98, a99
v_accvgpr_mov_b32 a97, a99
v_accvgpr_mov_b32 a96, a99
v_accvgpr_mov_b32 a131, a99
v_accvgpr_mov_b32 a130, a99
v_accvgpr_mov_b32 a129, a99
v_accvgpr_mov_b32 a128, a99
v_accvgpr_mov_b32 a135, a99
v_accvgpr_mov_b32 a134, a99
v_accvgpr_mov_b32 a133, a99
v_accvgpr_mov_b32 a132, a99
v_accvgpr_read_b32 v9, a99
v_accvgpr_read_b32 v8, a99
v_accvgpr_read_b32 v7, a99
v_accvgpr_read_b32 v6, a99
v_accvgpr_write_b32 a161, v9
v_accvgpr_write_b32 a160, v8
v_accvgpr_write_b32 a159, v7
v_accvgpr_write_b32 a158, v6
v_accvgpr_write_b32 a165, v9
v_accvgpr_write_b32 a164, v8
v_accvgpr_write_b32 a163, v7
v_accvgpr_write_b32 a162, v6
v_accvgpr_write_b32 a169, v9
v_accvgpr_write_b32 a168, v8
v_accvgpr_write_b32 a167, v7
v_accvgpr_write_b32 a166, v6
v_accvgpr_write_b32 a127, v9
v_accvgpr_write_b32 a126, v8
v_accvgpr_write_b32 a125, v7
v_accvgpr_write_b32 a124, v6
v_accvgpr_read_b32 v33, a99
v_accvgpr_read_b32 v32, a99
v_accvgpr_read_b32 v31, a99
v_accvgpr_read_b32 v30, a99
v_accvgpr_read_b32 v37, a99
v_accvgpr_read_b32 v36, a99
v_accvgpr_read_b32 v35, a99
v_accvgpr_read_b32 v34, a99
v_accvgpr_read_b32 v41, a99
v_accvgpr_read_b32 v40, a99
v_accvgpr_read_b32 v39, a99
v_accvgpr_read_b32 v38, a99
v_accvgpr_read_b32 v45, a99
v_accvgpr_read_b32 v44, a99
v_accvgpr_read_b32 v43, a99
v_accvgpr_read_b32 v42, a99
v_accvgpr_read_b32 v49, a99
v_accvgpr_read_b32 v48, a99
v_accvgpr_read_b32 v47, a99
v_accvgpr_read_b32 v46, a99
v_accvgpr_read_b32 v53, a99
v_accvgpr_read_b32 v52, a99
v_accvgpr_read_b32 v51, a99
v_accvgpr_read_b32 v50, a99
v_accvgpr_read_b32 v57, a99
v_accvgpr_read_b32 v56, a99
v_accvgpr_read_b32 v55, a99
v_accvgpr_read_b32 v54, a99
v_accvgpr_read_b32 v61, a99
v_accvgpr_read_b32 v60, a99
v_accvgpr_read_b32 v59, a99
v_accvgpr_read_b32 v58, a99
v_accvgpr_read_b32 v65, a99
v_accvgpr_read_b32 v64, a99
v_accvgpr_read_b32 v63, a99
v_accvgpr_read_b32 v62, a99
v_accvgpr_read_b32 v69, a99
v_accvgpr_read_b32 v68, a99
v_accvgpr_read_b32 v67, a99
v_accvgpr_read_b32 v66, a99
v_accvgpr_read_b32 v73, a99
v_accvgpr_read_b32 v72, a99
v_accvgpr_read_b32 v71, a99
v_accvgpr_read_b32 v70, a99
v_accvgpr_read_b32 v77, a99
v_accvgpr_read_b32 v76, a99
v_accvgpr_read_b32 v75, a99
v_accvgpr_read_b32 v74, a99
v_accvgpr_read_b32 v81, a99
v_accvgpr_read_b32 v80, a99
v_accvgpr_read_b32 v79, a99
v_accvgpr_read_b32 v78, a99
v_accvgpr_read_b32 v85, a99
v_accvgpr_read_b32 v84, a99
v_accvgpr_read_b32 v83, a99
v_accvgpr_read_b32 v82, a99
v_accvgpr_read_b32 v89, a99
v_accvgpr_read_b32 v88, a99
v_accvgpr_read_b32 v87, a99
v_accvgpr_read_b32 v86, a99
v_accvgpr_read_b32 v93, a99
v_accvgpr_read_b32 v92, a99
v_accvgpr_read_b32 v91, a99
v_accvgpr_read_b32 v90, a99
v_accvgpr_read_b32 v97, a99
v_accvgpr_read_b32 v96, a99
v_accvgpr_read_b32 v95, a99
v_accvgpr_read_b32 v94, a99
v_accvgpr_read_b32 v101, a99
v_accvgpr_read_b32 v100, a99
v_accvgpr_read_b32 v99, a99
v_accvgpr_read_b32 v98, a99
v_accvgpr_read_b32 v105, a99
v_accvgpr_read_b32 v104, a99
v_accvgpr_read_b32 v103, a99
v_accvgpr_read_b32 v102, a99
v_accvgpr_read_b32 v109, a99
v_accvgpr_read_b32 v108, a99
v_accvgpr_read_b32 v107, a99
v_accvgpr_read_b32 v106, a99
v_accvgpr_read_b32 v113, a99
v_accvgpr_read_b32 v112, a99
v_accvgpr_read_b32 v111, a99
v_accvgpr_read_b32 v110, a99
v_accvgpr_read_b32 v117, a99
v_accvgpr_read_b32 v116, a99
v_accvgpr_read_b32 v115, a99
v_accvgpr_read_b32 v114, a99
v_accvgpr_read_b32 v121, a99
v_accvgpr_read_b32 v120, a99
v_accvgpr_read_b32 v119, a99
v_accvgpr_read_b32 v118, a99
v_accvgpr_read_b32 v125, a99
v_accvgpr_read_b32 v124, a99
v_accvgpr_read_b32 v123, a99
v_accvgpr_read_b32 v122, a99
v_accvgpr_read_b32 v129, a99
v_accvgpr_read_b32 v128, a99
v_accvgpr_read_b32 v127, a99
v_accvgpr_read_b32 v126, a99
v_accvgpr_mov_b32 a229, a99
v_accvgpr_mov_b32 a228, a99
v_accvgpr_mov_b32 a227, a99
v_accvgpr_mov_b32 a226, a99
v_accvgpr_read_b32 v137, a99
v_accvgpr_read_b32 v136, a99
v_accvgpr_read_b32 v135, a99
v_accvgpr_read_b32 v134, a99
v_accvgpr_read_b32 v141, a99
v_accvgpr_read_b32 v140, a99
v_accvgpr_read_b32 v139, a99
v_accvgpr_read_b32 v138, a99
v_accvgpr_read_b32 v145, a99
v_accvgpr_read_b32 v144, a99
v_accvgpr_read_b32 v143, a99
v_accvgpr_read_b32 v142, a99
v_accvgpr_read_b32 v149, a99
v_accvgpr_read_b32 v148, a99
v_accvgpr_read_b32 v147, a99
v_accvgpr_read_b32 v146, a99
v_accvgpr_read_b32 v153, a99
v_accvgpr_read_b32 v152, a99
v_accvgpr_read_b32 v151, a99
v_accvgpr_read_b32 v150, a99
v_accvgpr_read_b32 v157, a99
v_accvgpr_read_b32 v156, a99
v_accvgpr_read_b32 v155, a99
v_accvgpr_read_b32 v154, a99
v_accvgpr_read_b32 v161, a99
v_accvgpr_read_b32 v160, a99
v_accvgpr_read_b32 v159, a99
v_accvgpr_read_b32 v158, a99
v_accvgpr_read_b32 v165, a99
v_accvgpr_read_b32 v164, a99
v_accvgpr_read_b32 v163, a99
v_accvgpr_read_b32 v162, a99
v_accvgpr_read_b32 v169, a99
v_accvgpr_read_b32 v168, a99
v_accvgpr_read_b32 v167, a99
v_accvgpr_read_b32 v166, a99
v_accvgpr_read_b32 v173, a99
v_accvgpr_read_b32 v172, a99
v_accvgpr_read_b32 v171, a99
v_accvgpr_read_b32 v170, a99
v_accvgpr_read_b32 v177, a99
v_accvgpr_read_b32 v176, a99
v_accvgpr_read_b32 v175, a99
v_accvgpr_read_b32 v174, a99
v_accvgpr_read_b32 v181, a99
v_accvgpr_read_b32 v180, a99
v_accvgpr_read_b32 v179, a99
v_accvgpr_read_b32 v178, a99
v_accvgpr_read_b32 v185, a99
v_accvgpr_read_b32 v184, a99
v_accvgpr_read_b32 v183, a99
v_accvgpr_read_b32 v182, a99
v_accvgpr_read_b32 v189, a99
v_accvgpr_read_b32 v188, a99
v_accvgpr_read_b32 v187, a99
v_accvgpr_read_b32 v186, a99
v_accvgpr_read_b32 v193, a99
v_accvgpr_read_b32 v192, a99
v_accvgpr_read_b32 v191, a99
v_accvgpr_read_b32 v190, a99
v_accvgpr_read_b32 v197, a99
v_accvgpr_read_b32 v196, a99
v_accvgpr_read_b32 v195, a99
v_accvgpr_read_b32 v194, a99
v_accvgpr_read_b32 v201, a99
v_accvgpr_read_b32 v200, a99
v_accvgpr_read_b32 v199, a99
v_accvgpr_read_b32 v198, a99
v_accvgpr_read_b32 v205, a99
v_accvgpr_read_b32 v204, a99
v_accvgpr_read_b32 v203, a99
v_accvgpr_read_b32 v202, a99
v_accvgpr_read_b32 v209, a99
v_accvgpr_read_b32 v208, a99
v_accvgpr_read_b32 v207, a99
v_accvgpr_read_b32 v206, a99
v_accvgpr_read_b32 v213, a99
v_accvgpr_read_b32 v212, a99
v_accvgpr_read_b32 v211, a99
v_accvgpr_read_b32 v210, a99
v_accvgpr_read_b32 v217, a99
v_accvgpr_read_b32 v216, a99
v_accvgpr_read_b32 v215, a99
v_accvgpr_read_b32 v214, a99
v_accvgpr_read_b32 v221, a99
v_accvgpr_read_b32 v220, a99
v_accvgpr_read_b32 v219, a99
v_accvgpr_read_b32 v218, a99
v_accvgpr_read_b32 v225, a99
v_accvgpr_read_b32 v224, a99
v_accvgpr_read_b32 v223, a99
v_accvgpr_read_b32 v222, a99
v_accvgpr_read_b32 v229, a99
v_accvgpr_read_b32 v228, a99
v_accvgpr_read_b32 v227, a99
v_accvgpr_read_b32 v226, a99
v_accvgpr_read_b32 v233, a99
v_accvgpr_read_b32 v232, a99
v_accvgpr_read_b32 v231, a99
v_accvgpr_read_b32 v230, a99
v_accvgpr_read_b32 v237, a99
v_accvgpr_read_b32 v236, a99
v_accvgpr_read_b32 v235, a99
v_accvgpr_read_b32 v234, a99
v_accvgpr_write_b32 a123, v9
v_accvgpr_write_b32 a122, v8
v_accvgpr_write_b32 a121, v7
v_accvgpr_write_b32 a120, v6
v_accvgpr_read_b32 v245, a99
v_accvgpr_read_b32 v244, a99
v_accvgpr_read_b32 v243, a99
v_accvgpr_read_b32 v242, a99
v_accvgpr_read_b32 v249, a99
v_accvgpr_read_b32 v248, a99
v_accvgpr_read_b32 v247, a99
v_accvgpr_read_b32 v246, a99
v_accvgpr_read_b32 v253, a99
v_accvgpr_read_b32 v252, a99
v_accvgpr_read_b32 v251, a99
v_accvgpr_read_b32 v250, a99
v_accvgpr_read_b32 v29, a99
v_accvgpr_read_b32 v28, a99
v_accvgpr_read_b32 v27, a99
v_accvgpr_read_b32 v26, a99
.LBB0_5:
v_accvgpr_read_b32 v130, a226
v_accvgpr_read_b32 v131, a227
v_accvgpr_read_b32 v132, a228
v_accvgpr_read_b32 v133, a229
v_accvgpr_mov_b32 a104, a132
v_accvgpr_mov_b32 a105, a133
v_accvgpr_mov_b32 a106, a134
v_accvgpr_mov_b32 a107, a135
v_accvgpr_mov_b32 a100, a128
v_accvgpr_mov_b32 a101, a129
v_accvgpr_mov_b32 a102, a130
v_accvgpr_mov_b32 a103, a131
v_lshrrev_b32_e32 v1, 4, v19
v_or_b32_e32 v2, 16, v1
v_or_b32_e32 v3, 32, v1
v_or_b32_e32 v4, 48, v1
v_or_b32_e32 v5, 64, v1
v_or_b32_e32 v6, 0x50, v1
s_movk_i32 s2, 0x60
v_or_b32_e32 v7, 0x60, v1
v_or_b32_e32 v8, 0x70, v1
v_or_b32_e32 v9, 0x80, v1
v_or_b32_e32 v10, 0x90, v1
v_or_b32_e32 v11, 0xa0, v1
v_or_b32_e32 v12, 0xb0, v1
v_or_b32_e32 v13, 0xc0, v1
v_or_b32_e32 v14, 0xd0, v1
v_or_b32_e32 v15, 0xe0, v1
v_or_b32_e32 v16, 0xf0, v1
v_lshlrev_b32_e32 v0, 3, v21
s_mul_i32 s0, s15, s13
s_ashr_i32 s1, s0, 31
s_lshl_b64 s[0:1], s[0:1], 1
s_add_u32 s3, s6, s0
s_addc_u32 s4, s7, s1
s_ashr_i32 s15, s14, 31
s_lshl_b64 s[0:1], s[14:15], 1
s_add_u32 s0, s3, s0
s_addc_u32 s1, s4, s1
v_mul_lo_u32 v1, v1, s13
v_mul_lo_u32 v2, v2, s13
v_mul_lo_u32 v3, v3, s13
v_mul_lo_u32 v4, v4, s13
v_mul_lo_u32 v5, v5, s13
v_mul_lo_u32 v255, v6, s13
v_mul_lo_u32 v254, v7, s13
v_mul_lo_u32 v6, v8, s13
v_accvgpr_write_b32 a136, v6
v_mul_lo_u32 v6, v9, s13
v_accvgpr_write_b32 a135, v6
v_mul_lo_u32 v6, v10, s13
v_accvgpr_write_b32 a134, v6
v_mul_lo_u32 v6, v11, s13
v_accvgpr_write_b32 a133, v6
v_mul_lo_u32 v6, v12, s13
v_accvgpr_write_b32 a132, v6
v_mul_lo_u32 v6, v13, s13
v_accvgpr_write_b32 a131, v6
v_mul_lo_u32 v6, v14, s13
v_accvgpr_write_b32 a130, v6
v_mul_lo_u32 v6, v15, s13
v_accvgpr_write_b32 a129, v6
v_mul_lo_u32 v6, v16, s13
v_accvgpr_write_b32 a128, v6
s_waitcnt lgkmcnt(7)
v_mfma_f32_16x16x32_f16 v[6:9], a[88:91], a[56:59], v[26:29]
s_waitcnt lgkmcnt(6)
v_mfma_f32_16x16x32_f16 v[6:9], a[92:95], a[60:63], v[6:9]
s_waitcnt lgkmcnt(5)
v_mfma_f32_16x16x32_f16 v[250:253], a[80:83], a[56:59], v[250:253]
s_waitcnt lgkmcnt(4)
v_mfma_f32_16x16x32_f16 v[250:253], a[84:87], a[60:63], v[250:253]
s_waitcnt lgkmcnt(3)
v_mfma_f32_16x16x32_f16 v[246:249], a[72:75], a[56:59], v[246:249]
s_waitcnt lgkmcnt(2)
v_mfma_f32_16x16x32_f16 v[246:249], a[76:79], a[60:63], v[246:249]
s_waitcnt lgkmcnt(1)
v_mfma_f32_16x16x32_f16 v[242:245], a[64:67], a[56:59], v[242:245]
s_waitcnt lgkmcnt(0)
v_mfma_f32_16x16x32_f16 v[242:245], a[68:71], a[60:63], v[242:245]
v_accvgpr_read_b32 v10, a120
v_accvgpr_read_b32 v11, a121
v_accvgpr_read_b32 v12, a122
v_accvgpr_read_b32 v13, a123
s_nop 1
v_mfma_f32_16x16x32_f16 v[238:241], a[88:91], a[48:51], v[10:13]
v_mfma_f32_16x16x32_f16 v[238:241], a[92:95], a[52:55], v[238:241]
v_mfma_f32_16x16x32_f16 v[234:237], a[80:83], a[48:51], v[234:237]
v_mfma_f32_16x16x32_f16 v[234:237], a[84:87], a[52:55], v[234:237]
v_mfma_f32_16x16x32_f16 v[230:233], a[72:75], a[48:51], v[230:233]
v_mfma_f32_16x16x32_f16 v[230:233], a[76:79], a[52:55], v[230:233]
v_mfma_f32_16x16x32_f16 v[226:229], a[64:67], a[48:51], v[226:229]
v_mfma_f32_16x16x32_f16 v[226:229], a[68:71], a[52:55], v[226:229]
v_mfma_f32_16x16x32_f16 v[222:225], a[88:91], a[40:43], v[222:225]
v_mfma_f32_16x16x32_f16 v[222:225], a[92:95], a[44:47], v[222:225]
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
v_mfma_f32_16x16x32_f16 v[190:193], a[88:91], a[24:27], v[190:193]
v_mfma_f32_16x16x32_f16 v[190:193], a[92:95], a[28:31], v[190:193]
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
v_mfma_f32_16x16x32_f16 v[158:161], a[88:91], a[8:11], v[158:161]
v_mfma_f32_16x16x32_f16 v[158:161], a[92:95], a[12:15], v[158:161]
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
v_mfma_f32_16x16x32_f16 v[130:133], a[64:67], a[0:3], v[130:133]
v_mfma_f32_16x16x32_f16 v[130:133], a[68:71], a[4:7], v[130:133]
s_waitcnt vmcnt(0), lgkmcnt(0)
s_barrier
v_add_u32_e32 v10, 0x18bc0, v17
ds_read_b128 a[64:67], v10
ds_read_b128 a[68:71], v10, offset:64
ds_read_b128 a[72:75], v10, offset:256
ds_read_b128 a[76:79], v10, offset:320
ds_read_b128 a[80:83], v10, offset:512
ds_read_b128 a[84:87], v10, offset:576
ds_read_b128 a[88:91], v10, offset:768
ds_read_b128 a[92:95], v10, offset:832
v_cvt_pk_f16_f32 v10, v6, v7
v_cvt_pk_f16_f32 v11, v8, v9
v_cvt_pk_f16_f32 v250, v250, v251
v_cvt_pk_f16_f32 v251, v252, v253
v_cvt_pk_f16_f32 v246, v246, v247
v_cvt_pk_f16_f32 v247, v248, v249
v_cvt_pk_f16_f32 v242, v242, v243
v_cvt_pk_f16_f32 v243, v244, v245
v_cvt_pk_f16_f32 v12, v238, v239
v_cvt_pk_f16_f32 v13, v240, v241
v_cvt_pk_f16_f32 v252, v234, v235
v_cvt_pk_f16_f32 v253, v236, v237
v_cvt_pk_f16_f32 v248, v230, v231
v_cvt_pk_f16_f32 v249, v232, v233
v_cvt_pk_f16_f32 v244, v226, v227
v_cvt_pk_f16_f32 v245, v228, v229
v_cvt_pk_f16_f32 v222, v222, v223
v_cvt_pk_f16_f32 v223, v224, v225
v_cvt_pk_f16_f32 v218, v218, v219
v_cvt_pk_f16_f32 v219, v220, v221
v_cvt_pk_f16_f32 v214, v214, v215
v_cvt_pk_f16_f32 v215, v216, v217
v_cvt_pk_f16_f32 v210, v210, v211
v_cvt_pk_f16_f32 v211, v212, v213
v_cvt_pk_f16_f32 v224, v206, v207
v_cvt_pk_f16_f32 v225, v208, v209
v_cvt_pk_f16_f32 v220, v202, v203
v_cvt_pk_f16_f32 v221, v204, v205
v_cvt_pk_f16_f32 v216, v198, v199
v_cvt_pk_f16_f32 v217, v200, v201
v_cvt_pk_f16_f32 v212, v194, v195
v_cvt_pk_f16_f32 v213, v196, v197
v_cvt_pk_f16_f32 v190, v190, v191
v_cvt_pk_f16_f32 v191, v192, v193
v_cvt_pk_f16_f32 v186, v186, v187
v_cvt_pk_f16_f32 v187, v188, v189
v_cvt_pk_f16_f32 v182, v182, v183
v_cvt_pk_f16_f32 v183, v184, v185
v_cvt_pk_f16_f32 v178, v178, v179
v_cvt_pk_f16_f32 v179, v180, v181
v_cvt_pk_f16_f32 v192, v174, v175
v_cvt_pk_f16_f32 v193, v176, v177
v_cvt_pk_f16_f32 v188, v170, v171
v_cvt_pk_f16_f32 v189, v172, v173
v_cvt_pk_f16_f32 v184, v166, v167
v_cvt_pk_f16_f32 v185, v168, v169
v_cvt_pk_f16_f32 v180, v162, v163
v_cvt_pk_f16_f32 v181, v164, v165
v_cvt_pk_f16_f32 v158, v158, v159
v_cvt_pk_f16_f32 v159, v160, v161
v_cvt_pk_f16_f32 v154, v154, v155
v_cvt_pk_f16_f32 v155, v156, v157
v_cvt_pk_f16_f32 v150, v150, v151
v_cvt_pk_f16_f32 v151, v152, v153
v_cvt_pk_f16_f32 v146, v146, v147
v_cvt_pk_f16_f32 v147, v148, v149
v_cvt_pk_f16_f32 v160, v142, v143
v_cvt_pk_f16_f32 v161, v144, v145
v_cvt_pk_f16_f32 v156, v138, v139
v_cvt_pk_f16_f32 v157, v140, v141
v_cvt_pk_f16_f32 v152, v134, v135
v_cvt_pk_f16_f32 v153, v136, v137
v_cvt_pk_f16_f32 v148, v130, v131
v_cvt_pk_f16_f32 v149, v132, v133
v_lshlrev_b32_e32 v6, 8, v18
v_and_b32_e32 v7, 0x70, v20
v_and_b32_e32 v14, 1, v18
v_lshlrev_b32_e32 v8, 12, v14
v_and_b32_e32 v15, 16, v18
v_lshlrev_b32_e32 v9, 4, v15
s_lshr_b32 s3, s12, 2
s_and_b32 s4, s22, 0x80
s_movk_i32 s5, 0x2e00
v_and_or_b32 v6, v6, s5, v8
v_or3_b32 v6, s4, v9, v6
v_mov_b32_e32 v8, 0x70
v_bitop3_b32 v16, s3, v20, v8, bitop3:0x78
v_or_b32_e32 v17, v6, v16
v_add_u32_e32 v8, 0, v17
ds_write_b128 v8, v[10:13]
v_xad_u32 v9, v17, 32, 0
ds_write_b128 v9, v[250:253]
v_xad_u32 v130, v17, 64, 0
ds_write_b128 v130, v[246:249]
v_bitop3_b32 v6, v6, s2, v16, bitop3:0x36
v_add_u32_e32 v131, 0, v6
ds_write_b128 v131, v[242:245]
s_waitcnt lgkmcnt(0)
s_barrier
v_and_b32_e32 v6, 0xe0, v19
v_lshlrev_b32_e32 v10, 4, v6
v_lshrrev_b32_e32 v6, 1, v6
v_lshlrev_b32_e32 v11, 8, v15
v_bitop3_b32 v6, v10, v6, v7, bitop3:0x36
v_lshl_add_u32 v7, v14, 13, 0
v_add3_u32 v132, v7, v11, v6
ds_read_b128 v[10:13], v132
ds_read_b128 v[16:19], v132, offset:256
ds_read_b128 v[20:23], v132, offset:128
ds_read_b128 v[138:141], v132, offset:384
s_waitcnt lgkmcnt(0)
s_barrier
ds_write_b128 v8, v[222:225]
ds_write_b128 v9, v[218:221]
ds_write_b128 v130, v[214:217]
ds_write_b128 v131, v[210:213]
s_waitcnt lgkmcnt(0)
s_barrier
ds_read_b128 v[142:145], v132
ds_read_b128 v[162:165], v132, offset:256
ds_read_b128 v[166:169], v132, offset:128
ds_read_b128 v[172:175], v132, offset:384
s_waitcnt lgkmcnt(0)
s_barrier
ds_write_b128 v8, v[190:193]
ds_write_b128 v9, v[186:189]
ds_write_b128 v130, v[182:185]
ds_write_b128 v131, v[178:181]
s_waitcnt lgkmcnt(0)
s_barrier
ds_read_b128 v[176:179], v132
ds_read_b128 v[182:185], v132, offset:256
ds_read_b128 v[186:189], v132, offset:128
ds_read_b128 v[192:195], v132, offset:384
s_waitcnt lgkmcnt(0)
s_barrier
ds_write_b128 v8, v[158:161]
ds_write_b128 v9, v[154:157]
ds_write_b128 v130, v[150:153]
ds_write_b128 v131, v[146:149]
s_waitcnt lgkmcnt(0)
s_barrier
ds_read_b128 v[148:151], v132
ds_read_b128 v[154:157], v132, offset:256
ds_read_b128 v[196:199], v132, offset:128
ds_read_b128 v[202:205], v132, offset:384
s_and_b32 s1, s1, 0xffff
s_mov_b32 s3, 0x27000
s_mov_b32 s2, 0x7ffffffe
v_mov_b32_e32 v14, v10
v_mov_b32_e32 v15, v11
v_add_lshl_u32 v133, v1, v0, 1
buffer_store_dwordx4 v[14:17], v133, s[0:3], 0, offen
v_mov_b32_e32 v136, v20
v_mov_b32_e32 v137, v21
v_add_lshl_u32 v134, v2, v0, 1
buffer_store_dwordx4 v[136:139], v134, s[0:3], 0, offen
v_mov_b32_e32 v14, v18
v_mov_b32_e32 v15, v19
v_add_lshl_u32 v135, v3, v0, 1
buffer_store_dwordx4 v[12:15], v135, s[0:3], 0, offen
v_mov_b32_e32 v24, v140
v_mov_b32_e32 v25, v141
v_add_lshl_u32 v136, v4, v0, 1
buffer_store_dwordx4 v[22:25], v136, s[0:3], 0, offen
v_mov_b32_e32 v160, v142
v_mov_b32_e32 v161, v143
v_add_lshl_u32 v137, v5, v0, 1
buffer_store_dwordx4 v[160:163], v137, s[0:3], 0, offen
v_mov_b32_e32 v170, v166
v_mov_b32_e32 v171, v167
v_add_lshl_u32 v138, v255, v0, 1
buffer_store_dwordx4 v[170:173], v138, s[0:3], 0, offen
v_mov_b32_e32 v146, v164
v_mov_b32_e32 v147, v165
v_add_lshl_u32 v139, v254, v0, 1
buffer_store_dwordx4 v[144:147], v139, s[0:3], 0, offen
v_mov_b32_e32 v170, v174
v_mov_b32_e32 v171, v175
v_accvgpr_read_b32 v1, a136
v_add_lshl_u32 v140, v1, v0, 1
buffer_store_dwordx4 v[168:171], v140, s[0:3], 0, offen
v_mov_b32_e32 v180, v176
v_mov_b32_e32 v181, v177
v_accvgpr_read_b32 v1, a135
v_add_lshl_u32 v141, v1, v0, 1
buffer_store_dwordx4 v[180:183], v141, s[0:3], 0, offen
v_mov_b32_e32 v190, v186
v_mov_b32_e32 v191, v187
v_accvgpr_read_b32 v1, a134
v_add_lshl_u32 v142, v1, v0, 1
buffer_store_dwordx4 v[190:193], v142, s[0:3], 0, offen
v_mov_b32_e32 v180, v184
v_mov_b32_e32 v181, v185
v_accvgpr_read_b32 v1, a133
v_add_lshl_u32 v143, v1, v0, 1
buffer_store_dwordx4 v[178:181], v143, s[0:3], 0, offen
v_mov_b32_e32 v190, v194
v_mov_b32_e32 v191, v195
v_accvgpr_read_b32 v1, a132
v_add_lshl_u32 v144, v1, v0, 1
buffer_store_dwordx4 v[188:191], v144, s[0:3], 0, offen
s_waitcnt lgkmcnt(3)
v_mov_b32_e32 v152, v148
v_mov_b32_e32 v153, v149
v_accvgpr_read_b32 v1, a131
v_add_lshl_u32 v145, v1, v0, 1
s_waitcnt lgkmcnt(2)
buffer_store_dwordx4 v[152:155], v145, s[0:3], 0, offen
s_waitcnt lgkmcnt(1)
v_mov_b32_e32 v200, v196
v_mov_b32_e32 v201, v197
v_accvgpr_read_b32 v1, a130
v_add_lshl_u32 v146, v1, v0, 1
s_waitcnt lgkmcnt(0)
buffer_store_dwordx4 v[200:203], v146, s[0:3], 0, offen
v_mov_b32_e32 v152, v156
v_mov_b32_e32 v153, v157
v_accvgpr_read_b32 v1, a129
v_add_lshl_u32 v147, v1, v0, 1
buffer_store_dwordx4 v[150:153], v147, s[0:3], 0, offen
v_mov_b32_e32 v200, v204
v_mov_b32_e32 v201, v205
v_accvgpr_read_b32 v1, a128
v_add_lshl_u32 v148, v1, v0, 1
buffer_store_dwordx4 v[198:201], v148, s[0:3], 0, offen
v_mfma_f32_16x16x32_f16 v[0:3], a[64:67], a[56:59], v[126:129]
v_mfma_f32_16x16x32_f16 v[0:3], a[68:71], a[60:63], v[0:3]
v_mfma_f32_16x16x32_f16 v[4:7], a[72:75], a[56:59], v[122:125]
v_mfma_f32_16x16x32_f16 v[4:7], a[76:79], a[60:63], v[4:7]
v_mfma_f32_16x16x32_f16 v[10:13], a[80:83], a[56:59], v[118:121]
v_mfma_f32_16x16x32_f16 v[10:13], a[84:87], a[60:63], v[10:13]
v_mfma_f32_16x16x32_f16 v[114:117], a[88:91], a[56:59], v[114:117]
v_mfma_f32_16x16x32_f16 v[114:117], a[92:95], a[60:63], v[114:117]
v_mfma_f32_16x16x32_f16 v[110:113], a[64:67], a[48:51], v[110:113]
v_mfma_f32_16x16x32_f16 v[110:113], a[68:71], a[52:55], v[110:113]
v_mfma_f32_16x16x32_f16 v[106:109], a[72:75], a[48:51], v[106:109]
v_mfma_f32_16x16x32_f16 v[106:109], a[76:79], a[52:55], v[106:109]
v_mfma_f32_16x16x32_f16 v[102:105], a[80:83], a[48:51], v[102:105]
v_mfma_f32_16x16x32_f16 v[102:105], a[84:87], a[52:55], v[102:105]
v_mfma_f32_16x16x32_f16 v[98:101], a[88:91], a[48:51], v[98:101]
v_mfma_f32_16x16x32_f16 v[98:101], a[92:95], a[52:55], v[98:101]
v_mfma_f32_16x16x32_f16 v[94:97], a[64:67], a[40:43], v[94:97]
v_mfma_f32_16x16x32_f16 v[94:97], a[68:71], a[44:47], v[94:97]
v_mfma_f32_16x16x32_f16 v[90:93], a[72:75], a[40:43], v[90:93]
v_mfma_f32_16x16x32_f16 v[90:93], a[76:79], a[44:47], v[90:93]
v_mfma_f32_16x16x32_f16 v[86:89], a[80:83], a[40:43], v[86:89]
v_mfma_f32_16x16x32_f16 v[86:89], a[84:87], a[44:47], v[86:89]
v_mfma_f32_16x16x32_f16 v[82:85], a[88:91], a[40:43], v[82:85]
v_mfma_f32_16x16x32_f16 v[82:85], a[92:95], a[44:47], v[82:85]
v_mfma_f32_16x16x32_f16 v[78:81], a[64:67], a[32:35], v[78:81]
v_mfma_f32_16x16x32_f16 v[78:81], a[68:71], a[36:39], v[78:81]
v_mfma_f32_16x16x32_f16 v[74:77], a[72:75], a[32:35], v[74:77]
v_mfma_f32_16x16x32_f16 v[74:77], a[76:79], a[36:39], v[74:77]
v_mfma_f32_16x16x32_f16 v[70:73], a[80:83], a[32:35], v[70:73]
v_mfma_f32_16x16x32_f16 v[70:73], a[84:87], a[36:39], v[70:73]
v_mfma_f32_16x16x32_f16 v[66:69], a[88:91], a[32:35], v[66:69]
v_mfma_f32_16x16x32_f16 v[66:69], a[92:95], a[36:39], v[66:69]
v_mfma_f32_16x16x32_f16 v[62:65], a[64:67], a[24:27], v[62:65]
v_mfma_f32_16x16x32_f16 v[62:65], a[68:71], a[28:31], v[62:65]
v_mfma_f32_16x16x32_f16 v[58:61], a[72:75], a[24:27], v[58:61]
v_mfma_f32_16x16x32_f16 v[58:61], a[76:79], a[28:31], v[58:61]
v_mfma_f32_16x16x32_f16 v[54:57], a[80:83], a[24:27], v[54:57]
v_mfma_f32_16x16x32_f16 v[54:57], a[84:87], a[28:31], v[54:57]
v_mfma_f32_16x16x32_f16 v[50:53], a[88:91], a[24:27], v[50:53]
v_mfma_f32_16x16x32_f16 v[50:53], a[92:95], a[28:31], v[50:53]
v_mfma_f32_16x16x32_f16 v[46:49], a[64:67], a[16:19], v[46:49]
v_mfma_f32_16x16x32_f16 v[46:49], a[68:71], a[20:23], v[46:49]
v_mfma_f32_16x16x32_f16 v[42:45], a[72:75], a[16:19], v[42:45]
v_mfma_f32_16x16x32_f16 v[42:45], a[76:79], a[20:23], v[42:45]
v_mfma_f32_16x16x32_f16 v[38:41], a[80:83], a[16:19], v[38:41]
v_mfma_f32_16x16x32_f16 v[38:41], a[84:87], a[20:23], v[38:41]
v_mfma_f32_16x16x32_f16 v[34:37], a[88:91], a[16:19], v[34:37]
v_mfma_f32_16x16x32_f16 v[34:37], a[92:95], a[20:23], v[34:37]
v_mfma_f32_16x16x32_f16 v[30:33], a[64:67], a[8:11], v[30:33]
v_mfma_f32_16x16x32_f16 v[30:33], a[68:71], a[12:15], v[30:33]
v_accvgpr_read_b32 v14, a124
v_accvgpr_read_b32 v15, a125
v_accvgpr_read_b32 v16, a126
v_accvgpr_read_b32 v17, a127
s_nop 1
v_mfma_f32_16x16x32_f16 v[26:29], a[72:75], a[8:11], v[14:17]
v_mfma_f32_16x16x32_f16 v[26:29], a[76:79], a[12:15], v[26:29]
s_nop 1
v_accvgpr_read_b32 v14, a166
v_accvgpr_read_b32 v15, a167
v_accvgpr_read_b32 v16, a168
v_accvgpr_read_b32 v17, a169
s_nop 1
v_mfma_f32_16x16x32_f16 v[22:25], a[80:83], a[8:11], v[14:17]
v_mfma_f32_16x16x32_f16 v[22:25], a[84:87], a[12:15], v[22:25]
s_nop 1
v_accvgpr_read_b32 v14, a162
v_accvgpr_read_b32 v15, a163
v_accvgpr_read_b32 v16, a164
v_accvgpr_read_b32 v17, a165
s_nop 1
v_mfma_f32_16x16x32_f16 v[18:21], a[88:91], a[8:11], v[14:17]
v_mfma_f32_16x16x32_f16 v[18:21], a[92:95], a[12:15], v[18:21]
s_nop 1
v_accvgpr_read_b32 v14, a158
v_accvgpr_read_b32 v15, a159
v_accvgpr_read_b32 v16, a160
v_accvgpr_read_b32 v17, a161
s_nop 1
v_mfma_f32_16x16x32_f16 v[14:17], a[64:67], a[0:3], v[14:17]
v_mfma_f32_16x16x32_f16 v[14:17], a[68:71], a[4:7], v[14:17]
v_accvgpr_read_b32 v121, a107
v_accvgpr_read_b32 v120, a106
v_accvgpr_read_b32 v119, a105
v_accvgpr_read_b32 v118, a104
s_nop 1
v_mfma_f32_16x16x32_f16 v[118:121], a[72:75], a[0:3], v[118:121]
v_mfma_f32_16x16x32_f16 v[118:121], a[76:79], a[4:7], v[118:121]
v_accvgpr_read_b32 v125, a103
v_accvgpr_read_b32 v124, a102
v_accvgpr_read_b32 v123, a101
v_accvgpr_read_b32 v122, a100
s_nop 1
v_mfma_f32_16x16x32_f16 v[122:125], a[80:83], a[0:3], v[122:125]
v_mfma_f32_16x16x32_f16 v[122:125], a[84:87], a[4:7], v[122:125]
v_accvgpr_read_b32 v129, a99
v_accvgpr_read_b32 v128, a98
v_accvgpr_read_b32 v127, a97
v_accvgpr_read_b32 v126, a96
s_nop 1
v_mfma_f32_16x16x32_f16 v[126:129], a[88:91], a[0:3], v[126:129]
v_mfma_f32_16x16x32_f16 v[126:129], a[92:95], a[4:7], v[126:129]
v_cvt_pk_f16_f32 v0, v0, v1
v_cvt_pk_f16_f32 v1, v2, v3
v_cvt_pk_f16_f32 v4, v4, v5
v_cvt_pk_f16_f32 v5, v6, v7
v_cvt_pk_f16_f32 v10, v10, v11
v_cvt_pk_f16_f32 v11, v12, v13
v_cvt_pk_f16_f32 v114, v114, v115
v_cvt_pk_f16_f32 v115, v116, v117
v_cvt_pk_f16_f32 v2, v110, v111
v_cvt_pk_f16_f32 v3, v112, v113
v_cvt_pk_f16_f32 v6, v106, v107
v_cvt_pk_f16_f32 v7, v108, v109
v_cvt_pk_f16_f32 v12, v102, v103
v_cvt_pk_f16_f32 v13, v104, v105
v_cvt_pk_f16_f32 v116, v98, v99
v_cvt_pk_f16_f32 v117, v100, v101
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
v_cvt_pk_f16_f32 v28, v118, v119
v_cvt_pk_f16_f32 v29, v120, v121
v_cvt_pk_f16_f32 v24, v122, v123
v_cvt_pk_f16_f32 v25, v124, v125
v_cvt_pk_f16_f32 v20, v126, v127
v_cvt_pk_f16_f32 v21, v128, v129
s_waitcnt lgkmcnt(0)
s_barrier
ds_write_b128 v8, v[0:3]
ds_write_b128 v9, v[4:7]
ds_write_b128 v130, v[10:13]
ds_write_b128 v131, v[114:117]
s_waitcnt lgkmcnt(0)
s_barrier
ds_read_b128 v[0:3], v132
ds_read_b128 v[10:13], v132, offset:256
ds_read_b128 v[4:7], v132, offset:128
ds_read_b128 v[34:37], v132, offset:384
s_waitcnt lgkmcnt(0)
s_barrier
ds_write_b128 v8, v[94:97]
ds_write_b128 v9, v[90:93]
ds_write_b128 v130, v[86:89]
ds_write_b128 v131, v[82:85]
s_waitcnt lgkmcnt(0)
s_barrier
ds_read_b128 v[14:17], v132
ds_read_b128 v[38:41], v132, offset:256
ds_read_b128 v[42:45], v132, offset:128
ds_read_b128 v[66:69], v132, offset:384
s_waitcnt lgkmcnt(0)
s_barrier
ds_write_b128 v8, v[62:65]
ds_write_b128 v9, v[58:61]
ds_write_b128 v130, v[54:57]
ds_write_b128 v131, v[50:53]
s_waitcnt lgkmcnt(0)
s_barrier
ds_read_b128 v[48:51], v132
ds_read_b128 v[54:57], v132, offset:256
ds_read_b128 v[58:61], v132, offset:128
ds_read_b128 v[70:73], v132, offset:384
s_waitcnt lgkmcnt(0)
s_barrier
ds_write_b128 v8, v[30:33]
ds_write_b128 v9, v[26:29]
ds_write_b128 v130, v[22:25]
ds_write_b128 v131, v[18:21]
s_waitcnt lgkmcnt(0)
s_barrier
ds_read_b128 v[20:23], v132
ds_read_b128 v[26:29], v132, offset:256
ds_read_b128 v[74:77], v132, offset:128
ds_read_b128 v[80:83], v132, offset:384
v_mov_b32_e32 v8, v0
v_mov_b32_e32 v9, v1
buffer_store_dwordx4 v[8:11], v133, s[0:3], 0, offen, offset:256
v_mov_b32_e32 v32, v4
v_mov_b32_e32 v33, v5
buffer_store_dwordx4 v[32:35], v134, s[0:3], 0, offen, offset:256
v_mov_b32_e32 v4, v12
v_mov_b32_e32 v5, v13
buffer_store_dwordx4 v[2:5], v135, s[0:3], 0, offen, offset:256
v_mov_b32_e32 v8, v36
v_mov_b32_e32 v9, v37
buffer_store_dwordx4 v[6:9], v136, s[0:3], 0, offen, offset:256
v_mov_b32_e32 v36, v14
v_mov_b32_e32 v37, v15
buffer_store_dwordx4 v[36:39], v137, s[0:3], 0, offen, offset:256
v_mov_b32_e32 v64, v42
v_mov_b32_e32 v65, v43
buffer_store_dwordx4 v[64:67], v138, s[0:3], 0, offen, offset:256
v_mov_b32_e32 v18, v40
v_mov_b32_e32 v19, v41
buffer_store_dwordx4 v[16:19], v139, s[0:3], 0, offen, offset:256
v_mov_b32_e32 v46, v68
v_mov_b32_e32 v47, v69
buffer_store_dwordx4 v[44:47], v140, s[0:3], 0, offen, offset:256
v_mov_b32_e32 v52, v48
v_mov_b32_e32 v53, v49
buffer_store_dwordx4 v[52:55], v141, s[0:3], 0, offen, offset:256
v_mov_b32_e32 v68, v58
v_mov_b32_e32 v69, v59
buffer_store_dwordx4 v[68:71], v142, s[0:3], 0, offen, offset:256
v_mov_b32_e32 v52, v56
v_mov_b32_e32 v53, v57
buffer_store_dwordx4 v[50:53], v143, s[0:3], 0, offen, offset:256
v_mov_b32_e32 v62, v72
v_mov_b32_e32 v63, v73
buffer_store_dwordx4 v[60:63], v144, s[0:3], 0, offen, offset:256
s_waitcnt lgkmcnt(3)
v_mov_b32_e32 v24, v20
v_mov_b32_e32 v25, v21
s_waitcnt lgkmcnt(2)
buffer_store_dwordx4 v[24:27], v145, s[0:3], 0, offen, offset:256
s_waitcnt lgkmcnt(1)
v_mov_b32_e32 v78, v74
v_mov_b32_e32 v79, v75
s_waitcnt lgkmcnt(0)
buffer_store_dwordx4 v[78:81], v146, s[0:3], 0, offen, offset:256
v_mov_b32_e32 v24, v28
v_mov_b32_e32 v25, v29
buffer_store_dwordx4 v[22:25], v147, s[0:3], 0, offen, offset:256
v_mov_b32_e32 v78, v82
v_mov_b32_e32 v79, v83
buffer_store_dwordx4 v[76:79], v148, s[0:3], 0, offen, offset:256
s_endpgm
	s_endpgm
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel v7_slice
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
		.amdhsa_next_free_sgpr 59
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
	.size	v7_slice, .Lfunc_end0-v7_slice
	.cfi_endproc
                                        ; -- End function
	.set v7_slice.num_vgpr, 256
	.set v7_slice.num_agpr, 242
	.set v7_slice.numbered_sgpr, 59
	.set v7_slice.num_named_barrier, 0
	.set v7_slice.private_seg_size, 0
	.set v7_slice.uses_vcc, 1
	.set v7_slice.uses_flat_scratch, 0
	.set v7_slice.has_dyn_sized_stack, 0
	.set v7_slice.has_recursion, 0
	.set v7_slice.has_indirect_call, 0
	.section	.AMDGPU.csdata,"",@progbits
; Kernel info:
; codeLenInByte = 14024
; TotalNumSgprs: 65
; NumVgprs: 256
; NumAgprs: 242
; TotalNumVgprs: 498
; ScratchSize: 0
; MemoryBound: 0
; FloatMode: 240
; IeeeMode: 1
; LDSByteSize: 0 bytes/workgroup (compile time only)
; SGPRBlocks: 8
; VGPRBlocks: 62
; NumSGPRsForWavesPerEU: 65
; NumVGPRsForWavesPerEU: 498
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
	.byte	5                               ; Abbreviation Code
	.byte	29                              ; DW_TAG_inlined_subroutine
	.byte	0                               ; DW_CHILDREN_no
	.byte	49                              ; DW_AT_abstract_origin
	.byte	19                              ; DW_FORM_ref4
	.byte	85                              ; DW_AT_ranges
	.byte	23                              ; DW_FORM_sec_offset
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
	.asciz	"/var/lib/jenkins/gfx9-gluon-tutorials/kernels/gemm/a16w16/v7_slice" ; string offset=24 ; /var/lib/jenkins/gfx9-gluon-tutorials/kernels/gemm/a16w16/v7_slice
.Linfo_string3:
	.asciz	"v7_slice"                      ; string offset=91 ; v7_slice
	.section	".note.GNU-stack","",@progbits
	.amdgpu_metadata
---
amdhsa.kernels:
  - .agpr_count:     242
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
    .name:           v7_slice
    .private_segment_fixed_size: 0
    .sgpr_count:     65
    .sgpr_spill_count: 0
    .symbol:         v7_slice.kd
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