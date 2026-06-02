	.amdgcn_target "amdgcn-amd-amdhsa--gfx950"
	.amdhsa_code_object_version 5
	.text
	.globl	v5_local_prefetch               ; -- Begin function v5_local_prefetch
	.p2align	8
	.type	v5_local_prefetch,@function
v5_local_prefetch:                      ; @v5_local_prefetch
.Lfunc_begin0:
	.cfi_sections .debug_frame
	.cfi_startproc
; %bb.9:
	.file	1 "kernels/gemm/a16w16/v5_local_prefetch" "matmul_kernel.py"
	s_load_dwordx8 s[8:15], s[4:5], 0x0
	s_waitcnt lgkmcnt(0)
	s_branch .LBB0_0
	.p2align	8
; %bb.10:
.LBB0_0:
	.file	2 "python/triton/language" "standard.py"
	s_add_i32 s0, s15, 0xff
	s_ashr_i32 s1, s0, 31
	s_lshr_b32 s1, s1, 24
	s_add_i32 s0, s0, s1
	s_ashr_i32 s0, s0, 8
	s_abs_i32 s1, s0
	v_cvt_f32_u32_e32 v1, s1
	s_sub_i32 s18, 0, s1
	s_abs_i32 s7, s16
	v_and_b32_e32 v2, 0x3ff, v0
	v_rcp_iflag_f32_e32 v1, v1
	v_readfirstlane_b32 s20, v2
	s_xor_b32 s6, s16, s0
	s_bfe_u32 s25, s20, 0x20006
	v_mul_f32_e32 v1, 0x4f7ffffe, v1
	v_cvt_u32_f32_e32 v1, v1
	s_ashr_i32 s6, s6, 31
	s_load_dwordx2 s[2:3], s[4:5], 0x20
	s_load_dword s17, s[4:5], 0x28
	v_readfirstlane_b32 s19, v1
	s_mul_i32 s18, s18, s19
	s_mul_hi_u32 s18, s19, s18
	s_add_i32 s19, s19, s18
	s_mul_hi_u32 s18, s7, s19
	s_mul_i32 s19, s18, s1
	s_sub_i32 s7, s7, s19
	s_add_i32 s19, s18, 1
	s_sub_i32 s21, s7, s1
	s_cmp_ge_u32 s7, s1
	s_cselect_b32 s18, s19, s18
	s_cselect_b32 s7, s21, s7
	s_add_i32 s19, s18, 1
	s_cmp_ge_u32 s7, s1
	s_cselect_b32 s1, s19, s18
	s_xor_b32 s1, s1, s6
	s_sub_i32 s1, s1, s6
	s_mul_i32 s0, s1, s0
	s_lshl_b32 s7, s1, 8
	s_sub_i32 s6, s16, s0
	s_waitcnt lgkmcnt(0)
	s_mul_i32 s0, s7, s3
	s_ashr_i32 s1, s0, 31
	s_lshl_b64 s[0:1], s[0:1], 1
	s_add_u32 s8, s8, s0
	s_addc_u32 s22, s9, s1
	s_lshl_b32 s6, s6, 8
	s_mul_i32 s0, s6, s17
	v_lshlrev_b32_e32 v1, 1, v2
	s_ashr_i32 s1, s0, 31
	v_and_b32_e32 v1, 0x70, v1
	s_lshl_b64 s[0:1], s[0:1], 1
	v_or_b32_e32 v1, s25, v1
	v_lshlrev_b32_e32 v2, 3, v2
	s_add_u32 s0, s10, s0
	s_mul_i32 s21, s25, 0x420
	v_or_b32_e32 v3, 4, v1
	v_and_b32_e32 v2, 56, v2
	s_addc_u32 s23, s11, s1
	v_mul_lo_u32 v10, v1, s3
	s_add_i32 s26, s21, 0
	v_or_b32_e32 v4, 8, v1
	v_mul_lo_u32 v11, v3, s3
	s_and_b32 s9, s22, 0xffff
	s_mov_b32 s11, 0x27000
	s_mov_b32 s10, 0x7ffffffe
	v_add_lshl_u32 v18, v10, v2, 1
	s_mov_b32 m0, s26
	v_or_b32_e32 v5, 12, v1
	v_mul_lo_u32 v12, v4, s3
	buffer_load_dwordx4 v18, s[8:11], 0 offen lds
	s_add_i32 m0, s26, 0x1080
	v_add_lshl_u32 v19, v11, v2, 1
	v_or_b32_e32 v6, 0x80, v1
	v_mul_lo_u32 v13, v5, s3
	buffer_load_dwordx4 v19, s[8:11], 0 offen lds
	s_add_i32 m0, s26, 0x2100
	v_add_lshl_u32 v20, v12, v2, 1
	v_or_b32_e32 v7, 0x84, v1
	v_mul_lo_u32 v14, v6, s3
	buffer_load_dwordx4 v20, s[8:11], 0 offen lds
	s_add_i32 m0, s26, 0x3180
	v_add_lshl_u32 v13, v13, v2, 1
	v_or_b32_e32 v8, 0x88, v1
	v_mul_lo_u32 v15, v7, s3
	buffer_load_dwordx4 v13, s[8:11], 0 offen lds
	s_add_i32 m0, s26, 0x4200
	v_add_lshl_u32 v14, v14, v2, 1
	v_or_b32_e32 v9, 0x8c, v1
	v_mul_lo_u32 v16, v8, s3
	buffer_load_dwordx4 v14, s[8:11], 0 offen lds
	s_add_i32 m0, s26, 0x5280
	v_add_lshl_u32 v15, v15, v2, 1
	v_mul_lo_u32 v17, v9, s3
	buffer_load_dwordx4 v15, s[8:11], 0 offen lds
	s_add_i32 m0, s26, 0x6300
	v_add_lshl_u32 v16, v16, v2, 1
	buffer_load_dwordx4 v16, s[8:11], 0 offen lds
	s_add_i32 m0, s26, 0x7380
	v_add_lshl_u32 v17, v17, v2, 1
	buffer_load_dwordx4 v17, s[8:11], 0 offen lds
	s_add_i32 s9, 0, 0x107e0
	v_mul_lo_u32 v1, v1, s17
	v_mul_lo_u32 v3, v3, s17
	v_mul_lo_u32 v4, v4, s17
	v_mul_lo_u32 v5, v5, s17
	v_mul_lo_u32 v6, v6, s17
	v_mul_lo_u32 v7, v7, s17
	v_mul_lo_u32 v8, v8, s17
	v_mul_lo_u32 v9, v9, s17
	s_add_i32 s17, s9, s21
	s_add_i32 s24, s2, 63
	s_and_b32 s1, s23, 0xffff
	s_mov_b32 s2, s10
	s_mov_b32 s3, s11
	v_add_lshl_u32 v10, v1, v2, 1
	s_mov_b32 m0, s17
	v_add_lshl_u32 v11, v3, v2, 1
	buffer_load_dwordx4 v10, s[0:3], 0 offen lds
	s_add_i32 m0, s17, 0x1080
	v_add_lshl_u32 v4, v4, v2, 1
	buffer_load_dwordx4 v11, s[0:3], 0 offen lds
	s_add_i32 m0, s17, 0x2100
	v_add_lshl_u32 v5, v5, v2, 1
	buffer_load_dwordx4 v4, s[0:3], 0 offen lds
	s_add_i32 m0, s17, 0x3180
	v_add_lshl_u32 v6, v6, v2, 1
	buffer_load_dwordx4 v5, s[0:3], 0 offen lds
	s_add_i32 m0, s17, 0x4200
	v_add_lshl_u32 v7, v7, v2, 1
	buffer_load_dwordx4 v6, s[0:3], 0 offen lds
	s_add_i32 m0, s17, 0x5280
	v_add_lshl_u32 v8, v8, v2, 1
	buffer_load_dwordx4 v7, s[0:3], 0 offen lds
	s_add_i32 m0, s17, 0x6300
	v_add_lshl_u32 v9, v9, v2, 1
	buffer_load_dwordx4 v8, s[0:3], 0 offen lds
	s_add_i32 m0, s17, 0x7380
	s_add_u32 s28, s8, 0x80
	buffer_load_dwordx4 v9, s[0:3], 0 offen lds
	s_addc_u32 s1, s22, 0
	s_add_u32 s16, s0, 0x80
	s_addc_u32 s2, s23, 0
	s_and_b32 s29, s1, 0xffff
	s_mov_b32 s30, s10
	s_mov_b32 s31, s11
	s_add_i32 m0, s17, 0xffff7c20
	; asyncmark
	s_barrier
	buffer_load_dwordx4 v18, s[28:31], 0 offen lds
	s_add_i32 m0, s17, 0xffff8ca0
	s_mov_b32 s18, s10
	buffer_load_dwordx4 v19, s[28:31], 0 offen lds
	s_add_i32 m0, s17, 0xffff9d20
	s_mov_b32 s19, s11
	buffer_load_dwordx4 v20, s[28:31], 0 offen lds
	s_add_i32 m0, s17, 0xffffada0
	v_and_b32_e32 v1, 63, v0
	buffer_load_dwordx4 v13, s[28:31], 0 offen lds
	s_add_i32 m0, s17, 0xffffbe20
	v_and_b32_e32 v3, 15, v0
	buffer_load_dwordx4 v14, s[28:31], 0 offen lds
	s_add_i32 m0, s17, 0xffffcea0
	v_lshl_or_b32 v2, s25, 6, v1
	buffer_load_dwordx4 v15, s[28:31], 0 offen lds
	s_add_i32 m0, s17, 0xffffdf20
	v_lshlrev_b32_e32 v1, 10, v3
	buffer_load_dwordx4 v16, s[28:31], 0 offen lds
	s_add_i32 m0, s17, 0xffffefa0
	s_and_b32 s17, s2, 0xffff
	buffer_load_dwordx4 v17, s[28:31], 0 offen lds
	s_add_i32 m0, s26, 0x18be0
	s_movk_i32 s1, 0xb0
	buffer_load_dwordx4 v10, s[16:19], 0 offen lds
	s_add_i32 m0, s26, 0x19c60
	v_accvgpr_write_b32 a128, v2
	buffer_load_dwordx4 v11, s[16:19], 0 offen lds
	s_add_i32 m0, s26, 0x1ace0
	v_and_or_b32 v2, v2, s1, v1
	buffer_load_dwordx4 v4, s[16:19], 0 offen lds
	s_add_i32 m0, s26, 0x1bd60
	s_lshl_b32 s1, s20, 1
	buffer_load_dwordx4 v5, s[16:19], 0 offen lds
	s_add_i32 m0, s26, 0x1cde0
	v_and_b32_e32 v0, 48, v0
	buffer_load_dwordx4 v6, s[16:19], 0 offen lds
	s_add_i32 m0, s26, 0x1de60
	s_and_b32 s1, s1, 0x80
	buffer_load_dwordx4 v7, s[16:19], 0 offen lds
	s_add_i32 m0, s26, 0x1eee0
	v_accvgpr_write_b32 a129, v3
	buffer_load_dwordx4 v8, s[16:19], 0 offen lds
	s_add_i32 m0, s26, 0x1ff60
	v_lshlrev_b32_e32 v3, 5, v3
	buffer_load_dwordx4 v9, s[16:19], 0 offen lds
	v_or3_b32 v0, v0, s1, v1
	v_add_u32_e32 v12, v2, v3
	v_add_u32_e32 v1, v0, v3
	v_add_u32_e32 v2, 0, v12
	v_add_u32_e32 v0, s9, v1
	; asyncmark
	; wait_asyncmark(1)
	s_waitcnt vmcnt(16)
	s_barrier
	ds_read_b128 a[120:123], v2
	ds_read_b128 a[124:127], v2 offset:64
	ds_read_b128 a[60:63], v2 offset:256
	ds_read_b128 a[36:39], v2 offset:320
	ds_read_b128 a[104:107], v2 offset:512
	ds_read_b128 a[108:111], v2 offset:576
	ds_read_b128 a[72:75], v2 offset:768
	ds_read_b128 a[76:79], v2 offset:832
	ds_read_b128 a[64:67], v2 offset:16896
	ds_read_b128 a[68:71], v2 offset:16960
	ds_read_b128 a[32:35], v2 offset:17152
	ds_read_b128 a[20:23], v2 offset:17216
	ds_read_b128 a[40:43], v2 offset:17408
	ds_read_b128 a[4:7], v2 offset:17472
	ds_read_b128 a[48:51], v2 offset:17664
	ds_read_b128 a[0:3], v2 offset:17728
	ds_read_b128 a[96:99], v0
	ds_read_b128 a[100:103], v0 offset:64
	ds_read_b128 a[88:91], v0 offset:256
	ds_read_b128 a[92:95], v0 offset:320
	ds_read_b128 a[56:59], v0 offset:512
	ds_read_b128 a[12:15], v0 offset:576
	ds_read_b128 a[80:83], v0 offset:768
	ds_read_b128 a[84:87], v0 offset:832
	ds_read_b128 a[52:55], v0 offset:16896
	ds_read_b128 a[16:19], v0 offset:16960
	ds_read_b128 a[112:115], v0 offset:17152
	ds_read_b128 a[116:119], v0 offset:17216
	ds_read_b128 a[28:31], v0 offset:17408
	ds_read_b128 a[24:27], v0 offset:17472
	ds_read_b128 a[44:47], v0 offset:17664
	ds_read_b128 a[8:11], v0 offset:17728
	s_cmpk_lt_i32 s24, 0x80
	s_mov_b32 s3, 0
	s_cbranch_scc1 .LBB0_5
; %bb.1:                                ; %.lr.ph
	s_lshr_b32 s10, s24, 6
	s_add_u32 s0, s0, 0x100
	s_addc_u32 s1, s23, 0
	s_add_u32 s8, s8, 0x100
	s_addc_u32 s9, s22, 0
	s_add_i32 s2, s10, -2
	v_accvgpr_write_b32 a147, v17
	v_accvgpr_write_b32 a146, v16
	v_accvgpr_write_b32 a145, v15
	v_accvgpr_write_b32 a144, v14
	v_accvgpr_write_b32 a143, v13
	v_accvgpr_write_b32 a142, v20
	v_accvgpr_write_b32 a141, v19
	v_accvgpr_write_b32 a140, v18
	v_accvgpr_write_b32 a139, v12
	v_accvgpr_write_b32 a138, v9
	v_accvgpr_write_b32 a137, v8
	v_accvgpr_write_b32 a132, v11
	v_accvgpr_write_b32 a131, v10
	v_accvgpr_write_b32 a130, v1
	s_cmp_eq_u32 s10, 2
	v_accvgpr_write_b32 a134, v5
	v_accvgpr_write_b32 a135, v6
	v_accvgpr_write_b32 a136, v7
	s_cbranch_scc1 .LBB0_6
; %bb.2:                                ; %.lr.ph.split
	v_mov_b32_e32 v248, 0
	v_mov_b32_e32 v3, v248
	v_mov_b32_e32 v0, v248
	v_mov_b32_e32 v1, v248
	v_mov_b32_e32 v2, v248
	v_accvgpr_write_b32 a151, v3
	s_mov_b32 s19, 0x27000
	s_mov_b32 s18, 0x7ffffffe
	v_mov_b32_e32 v249, v248
	v_mov_b32_e32 v250, v248
	v_mov_b32_e32 v251, v248
	v_mov_b32_e32 v252, v248
	v_mov_b32_e32 v253, v248
	v_mov_b32_e32 v254, v248
	v_mov_b32_e32 v255, v248
	v_mov_b32_e32 v244, v248
	v_mov_b32_e32 v245, v248
	v_mov_b32_e32 v246, v248
	v_mov_b32_e32 v247, v248
	v_mov_b32_e32 v240, v248
	v_mov_b32_e32 v241, v248
	v_mov_b32_e32 v242, v248
	v_mov_b32_e32 v243, v248
	v_accvgpr_write_b32 a166, v248
	v_accvgpr_write_b32 a167, v248
	v_accvgpr_write_b32 a168, v248
	v_accvgpr_write_b32 a169, v248
	v_mov_b32_e32 v232, v248
	v_mov_b32_e32 v233, v248
	v_mov_b32_e32 v234, v248
	v_mov_b32_e32 v235, v248
	v_mov_b32_e32 v228, v248
	v_mov_b32_e32 v229, v248
	v_mov_b32_e32 v230, v248
	v_mov_b32_e32 v231, v248
	v_accvgpr_write_b32 a188, v248
	v_accvgpr_write_b32 a189, v248
	v_accvgpr_write_b32 a190, v248
	v_accvgpr_write_b32 a191, v248
	v_mov_b32_e32 v220, v248
	v_mov_b32_e32 v221, v248
	v_mov_b32_e32 v222, v248
	v_mov_b32_e32 v223, v248
	v_mov_b32_e32 v216, v248
	v_mov_b32_e32 v217, v248
	v_mov_b32_e32 v218, v248
	v_mov_b32_e32 v219, v248
	v_mov_b32_e32 v212, v248
	v_mov_b32_e32 v213, v248
	v_mov_b32_e32 v214, v248
	v_mov_b32_e32 v215, v248
	v_mov_b32_e32 v208, v248
	v_mov_b32_e32 v209, v248
	v_mov_b32_e32 v210, v248
	v_mov_b32_e32 v211, v248
	v_mov_b32_e32 v204, v248
	v_mov_b32_e32 v205, v248
	v_mov_b32_e32 v206, v248
	v_mov_b32_e32 v207, v248
	v_mov_b32_e32 v200, v248
	v_mov_b32_e32 v201, v248
	v_mov_b32_e32 v202, v248
	v_mov_b32_e32 v203, v248
	v_accvgpr_write_b32 a172, v248
	v_accvgpr_write_b32 a173, v248
	v_accvgpr_write_b32 a174, v248
	v_accvgpr_write_b32 a175, v248
	v_mov_b32_e32 v192, v248
	v_mov_b32_e32 v193, v248
	v_mov_b32_e32 v194, v248
	v_mov_b32_e32 v195, v248
	v_mov_b32_e32 v188, v248
	v_mov_b32_e32 v189, v248
	v_mov_b32_e32 v190, v248
	v_mov_b32_e32 v191, v248
	v_mov_b32_e32 v184, v248
	v_mov_b32_e32 v185, v248
	v_mov_b32_e32 v186, v248
	v_mov_b32_e32 v187, v248
	v_mov_b32_e32 v180, v248
	v_mov_b32_e32 v181, v248
	v_mov_b32_e32 v182, v248
	v_mov_b32_e32 v183, v248
	v_mov_b32_e32 v176, v248
	v_mov_b32_e32 v177, v248
	v_mov_b32_e32 v178, v248
	v_mov_b32_e32 v179, v248
	v_mov_b32_e32 v172, v248
	v_mov_b32_e32 v173, v248
	v_mov_b32_e32 v174, v248
	v_mov_b32_e32 v175, v248
	v_mov_b32_e32 v168, v248
	v_mov_b32_e32 v169, v248
	v_mov_b32_e32 v170, v248
	v_mov_b32_e32 v171, v248
	v_mov_b32_e32 v164, v248
	v_mov_b32_e32 v165, v248
	v_mov_b32_e32 v166, v248
	v_mov_b32_e32 v167, v248
	v_mov_b32_e32 v160, v248
	v_mov_b32_e32 v161, v248
	v_mov_b32_e32 v162, v248
	v_mov_b32_e32 v163, v248
	v_mov_b32_e32 v156, v248
	v_mov_b32_e32 v157, v248
	v_mov_b32_e32 v158, v248
	v_mov_b32_e32 v159, v248
	v_mov_b32_e32 v152, v248
	v_mov_b32_e32 v153, v248
	v_mov_b32_e32 v154, v248
	v_mov_b32_e32 v155, v248
	v_accvgpr_write_b32 a184, v248
	v_accvgpr_write_b32 a185, v248
	v_accvgpr_write_b32 a186, v248
	v_accvgpr_write_b32 a187, v248
	v_mov_b32_e32 v144, v248
	v_mov_b32_e32 v145, v248
	v_mov_b32_e32 v146, v248
	v_mov_b32_e32 v147, v248
	v_mov_b32_e32 v140, v248
	v_mov_b32_e32 v141, v248
	v_mov_b32_e32 v142, v248
	v_mov_b32_e32 v143, v248
	v_mov_b32_e32 v136, v248
	v_mov_b32_e32 v137, v248
	v_mov_b32_e32 v138, v248
	v_mov_b32_e32 v139, v248
	v_mov_b32_e32 v132, v248
	v_mov_b32_e32 v133, v248
	v_mov_b32_e32 v134, v248
	v_mov_b32_e32 v135, v248
	v_accvgpr_write_b32 a176, v248
	v_accvgpr_write_b32 a177, v248
	v_accvgpr_write_b32 a178, v248
	v_accvgpr_write_b32 a179, v248
	v_mov_b32_e32 v124, v248
	v_mov_b32_e32 v125, v248
	v_mov_b32_e32 v126, v248
	v_mov_b32_e32 v127, v248
	v_mov_b32_e32 v120, v248
	v_mov_b32_e32 v121, v248
	v_mov_b32_e32 v122, v248
	v_mov_b32_e32 v123, v248
	v_mov_b32_e32 v116, v248
	v_mov_b32_e32 v117, v248
	v_mov_b32_e32 v118, v248
	v_mov_b32_e32 v119, v248
	v_mov_b32_e32 v112, v248
	v_mov_b32_e32 v113, v248
	v_mov_b32_e32 v114, v248
	v_mov_b32_e32 v115, v248
	v_accvgpr_write_b32 a180, v248
	v_accvgpr_write_b32 a181, v248
	v_accvgpr_write_b32 a182, v248
	v_accvgpr_write_b32 a183, v248
	v_mov_b32_e32 v104, v248
	v_mov_b32_e32 v105, v248
	v_mov_b32_e32 v106, v248
	v_mov_b32_e32 v107, v248
	v_mov_b32_e32 v100, v248
	v_mov_b32_e32 v101, v248
	v_mov_b32_e32 v102, v248
	v_mov_b32_e32 v103, v248
	v_mov_b32_e32 v96, v248
	v_mov_b32_e32 v97, v248
	v_mov_b32_e32 v98, v248
	v_mov_b32_e32 v99, v248
	v_mov_b32_e32 v92, v248
	v_mov_b32_e32 v93, v248
	v_mov_b32_e32 v94, v248
	v_mov_b32_e32 v95, v248
	v_mov_b32_e32 v88, v248
	v_mov_b32_e32 v89, v248
	v_mov_b32_e32 v90, v248
	v_mov_b32_e32 v91, v248
	v_mov_b32_e32 v84, v248
	v_mov_b32_e32 v85, v248
	v_mov_b32_e32 v86, v248
	v_mov_b32_e32 v87, v248
	v_mov_b32_e32 v80, v248
	v_mov_b32_e32 v81, v248
	v_mov_b32_e32 v82, v248
	v_mov_b32_e32 v83, v248
	v_mov_b32_e32 v76, v248
	v_mov_b32_e32 v77, v248
	v_mov_b32_e32 v78, v248
	v_mov_b32_e32 v79, v248
	v_mov_b32_e32 v72, v248
	v_mov_b32_e32 v73, v248
	v_mov_b32_e32 v74, v248
	v_mov_b32_e32 v75, v248
	v_mov_b32_e32 v68, v248
	v_mov_b32_e32 v69, v248
	v_mov_b32_e32 v70, v248
	v_mov_b32_e32 v71, v248
	v_mov_b32_e32 v64, v248
	v_mov_b32_e32 v65, v248
	v_mov_b32_e32 v66, v248
	v_mov_b32_e32 v67, v248
	v_mov_b32_e32 v60, v248
	v_mov_b32_e32 v61, v248
	v_mov_b32_e32 v62, v248
	v_mov_b32_e32 v63, v248
	v_mov_b32_e32 v56, v248
	v_mov_b32_e32 v57, v248
	v_mov_b32_e32 v58, v248
	v_mov_b32_e32 v59, v248
	v_mov_b32_e32 v52, v248
	v_mov_b32_e32 v53, v248
	v_mov_b32_e32 v54, v248
	v_mov_b32_e32 v55, v248
	v_mov_b32_e32 v48, v248
	v_mov_b32_e32 v49, v248
	v_mov_b32_e32 v50, v248
	v_mov_b32_e32 v51, v248
	v_mov_b32_e32 v44, v248
	v_mov_b32_e32 v45, v248
	v_mov_b32_e32 v46, v248
	v_mov_b32_e32 v47, v248
	v_mov_b32_e32 v40, v248
	v_mov_b32_e32 v41, v248
	v_mov_b32_e32 v42, v248
	v_mov_b32_e32 v43, v248
	v_mov_b32_e32 v36, v248
	v_mov_b32_e32 v37, v248
	v_mov_b32_e32 v38, v248
	v_mov_b32_e32 v39, v248
	v_mov_b32_e32 v32, v248
	v_mov_b32_e32 v33, v248
	v_mov_b32_e32 v34, v248
	v_mov_b32_e32 v35, v248
	v_mov_b32_e32 v28, v248
	v_mov_b32_e32 v29, v248
	v_mov_b32_e32 v30, v248
	v_mov_b32_e32 v31, v248
	v_mov_b32_e32 v24, v248
	v_mov_b32_e32 v25, v248
	v_mov_b32_e32 v26, v248
	v_mov_b32_e32 v27, v248
	v_mov_b32_e32 v20, v248
	v_mov_b32_e32 v21, v248
	v_mov_b32_e32 v22, v248
	v_mov_b32_e32 v23, v248
	v_mov_b32_e32 v16, v248
	v_mov_b32_e32 v17, v248
	v_mov_b32_e32 v18, v248
	v_mov_b32_e32 v19, v248
	v_mov_b32_e32 v12, v248
	v_mov_b32_e32 v13, v248
	v_mov_b32_e32 v14, v248
	v_mov_b32_e32 v15, v248
	v_mov_b32_e32 v8, v248
	v_mov_b32_e32 v9, v248
	v_mov_b32_e32 v10, v248
	v_mov_b32_e32 v11, v248
	v_accvgpr_write_b32 a150, v2
	v_accvgpr_write_b32 a149, v1
	v_accvgpr_write_b32 a148, v0
	v_accvgpr_write_b32 a192, v248
	v_accvgpr_write_b32 a193, v248
	v_accvgpr_write_b32 a194, v248
	v_accvgpr_write_b32 a195, v248
	v_accvgpr_mov_b32 a133, a130
	v_accvgpr_mov_b32 a152, a131
	v_accvgpr_mov_b32 a160, a132
	v_accvgpr_write_b32 a159, v4
	v_accvgpr_mov_b32 a153, a137
	v_accvgpr_mov_b32 a161, a138
	v_accvgpr_mov_b32 a154, a139
	v_accvgpr_mov_b32 a162, a140
	v_accvgpr_mov_b32 a155, a141
	v_accvgpr_mov_b32 a163, a142
	v_accvgpr_mov_b32 a156, a143
	v_accvgpr_mov_b32 a164, a144
	v_accvgpr_mov_b32 a157, a145
	v_accvgpr_mov_b32 a165, a146
	v_accvgpr_mov_b32 a158, a147
.LBB0_3:                                ; =>This Inner Loop Header: Depth=1
	s_and_b32 s10, s3, 1
	s_mul_i32 s11, s10, 0x8400
	s_add_i32 s11, s11, 0
	s_add_i32 s11, s11, s21
	s_and_b32 s17, s9, 0xffff
	s_mov_b32 s16, s8
	s_mov_b32 m0, s11
	v_accvgpr_read_b32 v7, a162
	; wait_asyncmark(0)
	s_waitcnt vmcnt(0) lgkmcnt(0)
	s_barrier
	buffer_load_dwordx4 v7, s[16:19], 0 offen lds
	s_add_i32 m0, s11, 0x1080
	v_accvgpr_read_b32 v7, a155
	buffer_load_dwordx4 v7, s[16:19], 0 offen lds
	s_add_i32 m0, s11, 0x2100
	v_accvgpr_read_b32 v7, a163
	buffer_load_dwordx4 v7, s[16:19], 0 offen lds
	s_add_i32 m0, s11, 0x3180
	v_accvgpr_read_b32 v7, a156
	buffer_load_dwordx4 v7, s[16:19], 0 offen lds
	s_add_i32 m0, s11, 0x4200
	v_accvgpr_read_b32 v7, a164
	buffer_load_dwordx4 v7, s[16:19], 0 offen lds
	s_add_i32 m0, s11, 0x5280
	v_accvgpr_read_b32 v7, a157
	buffer_load_dwordx4 v7, s[16:19], 0 offen lds
	s_add_i32 m0, s11, 0x6300
	v_accvgpr_read_b32 v7, a165
	buffer_load_dwordx4 v7, s[16:19], 0 offen lds
	s_add_i32 m0, s11, 0x7380
	v_accvgpr_read_b32 v7, a158
	buffer_load_dwordx4 v7, s[16:19], 0 offen lds
	s_and_b32 s17, s1, 0xffff
	s_mov_b32 s16, s0
	s_add_i32 m0, s11, 0x107e0
	v_accvgpr_read_b32 v7, a152
	buffer_load_dwordx4 v7, s[16:19], 0 offen lds
	s_add_i32 m0, s11, 0x11860
	v_accvgpr_read_b32 v7, a160
	buffer_load_dwordx4 v7, s[16:19], 0 offen lds
	s_add_i32 m0, s11, 0x128e0
	v_accvgpr_read_b32 v7, a159
	v_accvgpr_read_b32 v4, a134
	buffer_load_dwordx4 v7, s[16:19], 0 offen lds
	s_add_i32 m0, s11, 0x13960
	v_accvgpr_read_b32 v5, a135
	buffer_load_dwordx4 v4, s[16:19], 0 offen lds
	s_add_i32 m0, s11, 0x149e0
	v_accvgpr_read_b32 v6, a136
	buffer_load_dwordx4 v5, s[16:19], 0 offen lds
	s_add_i32 m0, s11, 0x15a60
	v_accvgpr_read_b32 v4, a153
	buffer_load_dwordx4 v6, s[16:19], 0 offen lds
	s_add_i32 m0, s11, 0x16ae0
	v_mfma_f32_16x16x32_f16 v[248:251], a[96:99], a[120:123], v[248:251]
	buffer_load_dwordx4 v4, s[16:19], 0 offen lds
	s_add_i32 m0, s11, 0x17b60
	v_accvgpr_read_b32 v4, a161
	buffer_load_dwordx4 v4, s[16:19], 0 offen lds
	v_mfma_f32_16x16x32_f16 v[252:255], a[88:91], a[120:123], v[252:255]
	s_lshl_b32 s10, s10, 14
	s_xor_b32 s10, s10, 0x4000
	s_lshr_b32 s11, s10, 5
	v_mfma_f32_16x16x32_f16 v[244:247], a[56:59], a[120:123], v[244:247]
	s_or_b32 s10, s11, s10
	s_lshl1_add_u32 s10, s10, 0
	v_accvgpr_read_b32 v4, a154
	v_mfma_f32_16x16x32_f16 v[240:243], a[80:83], a[120:123], v[240:243]
	v_add_u32_e32 v4, s10, v4
	; asyncmark
	v_mfma_f32_16x16x32_f16 a[166:169], a[52:55], a[120:123], a[166:169]
	s_barrier
	s_add_u32 s8, s8, 0x80
	v_mfma_f32_16x16x32_f16 v[232:235], a[112:115], a[120:123], v[232:235]
	s_addc_u32 s9, s9, 0
	s_add_u32 s0, s0, 0x80
	s_addc_u32 s1, s1, 0
	v_mfma_f32_16x16x32_f16 v[228:231], a[28:31], a[120:123], v[228:231]
	s_add_i32 s3, s3, 1
	s_cmp_lg_u32 s2, s3
	v_mfma_f32_16x16x32_f16 a[120:123], a[44:47], a[120:123], a[188:191]
	v_mfma_f32_16x16x32_f16 a[188:191], a[8:11], a[124:127], a[120:123]
	v_mfma_f32_16x16x32_f16 v[220:223], a[96:99], a[60:63], v[220:223]
	v_mfma_f32_16x16x32_f16 v[216:219], a[88:91], a[60:63], v[216:219]
	v_mfma_f32_16x16x32_f16 v[212:215], a[56:59], a[60:63], v[212:215]
	v_mfma_f32_16x16x32_f16 v[208:211], a[80:83], a[60:63], v[208:211]
	v_mfma_f32_16x16x32_f16 v[204:207], a[52:55], a[60:63], v[204:207]
	v_mfma_f32_16x16x32_f16 v[200:203], a[112:115], a[60:63], v[200:203]
	v_mfma_f32_16x16x32_f16 a[120:123], a[28:31], a[60:63], a[172:175]
	v_mfma_f32_16x16x32_f16 v[192:195], a[44:47], a[60:63], v[192:195]
	v_mfma_f32_16x16x32_f16 v[220:223], a[100:103], a[36:39], v[220:223]
	v_mfma_f32_16x16x32_f16 v[216:219], a[92:95], a[36:39], v[216:219]
	v_mfma_f32_16x16x32_f16 v[212:215], a[12:15], a[36:39], v[212:215]
	v_mfma_f32_16x16x32_f16 v[208:211], a[84:87], a[36:39], v[208:211]
	v_mfma_f32_16x16x32_f16 v[204:207], a[16:19], a[36:39], v[204:207]
	v_mfma_f32_16x16x32_f16 v[200:203], a[116:119], a[36:39], v[200:203]
	v_mfma_f32_16x16x32_f16 a[172:175], a[24:27], a[36:39], a[120:123]
	v_mfma_f32_16x16x32_f16 v[192:195], a[8:11], a[36:39], v[192:195]
	v_mfma_f32_16x16x32_f16 a[36:39], a[56:59], a[72:75], a[184:187]
	v_mfma_f32_16x16x32_f16 a[184:187], a[12:15], a[76:79], a[36:39]
	v_mfma_f32_16x16x32_f16 a[36:39], a[44:47], a[72:75], a[176:179]
	v_mfma_f32_16x16x32_f16 v[60:63], a[96:99], a[40:43], v[60:63]
	v_mfma_f32_16x16x32_f16 v[56:59], a[88:91], a[40:43], v[56:59]
	v_mfma_f32_16x16x32_f16 v[52:55], a[56:59], a[40:43], v[52:55]
	v_mfma_f32_16x16x32_f16 v[48:51], a[80:83], a[40:43], v[48:51]
	v_mfma_f32_16x16x32_f16 v[44:47], a[52:55], a[40:43], v[44:47]
	v_mfma_f32_16x16x32_f16 v[40:43], a[112:115], a[40:43], v[40:43]
	v_mfma_f32_16x16x32_f16 v[36:39], a[28:31], a[40:43], v[36:39]
	v_mfma_f32_16x16x32_f16 v[32:35], a[44:47], a[40:43], v[32:35]
	v_mfma_f32_16x16x32_f16 v[188:191], a[96:99], a[104:107], v[188:191]
	v_mfma_f32_16x16x32_f16 v[184:187], a[88:91], a[104:107], v[184:187]
	v_mfma_f32_16x16x32_f16 v[180:183], a[56:59], a[104:107], v[180:183]
	v_mfma_f32_16x16x32_f16 v[176:179], a[80:83], a[104:107], v[176:179]
	v_mfma_f32_16x16x32_f16 v[172:175], a[52:55], a[104:107], v[172:175]
	v_mfma_f32_16x16x32_f16 v[168:171], a[112:115], a[104:107], v[168:171]
	v_mfma_f32_16x16x32_f16 v[164:167], a[28:31], a[104:107], v[164:167]
	v_mfma_f32_16x16x32_f16 v[160:163], a[44:47], a[104:107], v[160:163]
	v_mfma_f32_16x16x32_f16 v[156:159], a[96:99], a[72:75], v[156:159]
	v_mfma_f32_16x16x32_f16 v[152:155], a[88:91], a[72:75], v[152:155]
	v_mfma_f32_16x16x32_f16 v[144:147], a[80:83], a[72:75], v[144:147]
	v_mfma_f32_16x16x32_f16 v[140:143], a[52:55], a[72:75], v[140:143]
	v_mfma_f32_16x16x32_f16 v[136:139], a[112:115], a[72:75], v[136:139]
	v_mfma_f32_16x16x32_f16 v[132:135], a[28:31], a[72:75], v[132:135]
	v_mfma_f32_16x16x32_f16 a[176:179], a[8:11], a[76:79], a[36:39]
	v_mfma_f32_16x16x32_f16 v[124:127], a[96:99], a[64:67], v[124:127]
	v_mfma_f32_16x16x32_f16 v[120:123], a[88:91], a[64:67], v[120:123]
	v_mfma_f32_16x16x32_f16 v[116:119], a[56:59], a[64:67], v[116:119]
	v_mfma_f32_16x16x32_f16 v[112:115], a[80:83], a[64:67], v[112:115]
	v_mfma_f32_16x16x32_f16 a[36:39], a[52:55], a[64:67], a[180:183]
	v_mfma_f32_16x16x32_f16 v[104:107], a[112:115], a[64:67], v[104:107]
	v_mfma_f32_16x16x32_f16 v[100:103], a[28:31], a[64:67], v[100:103]
	v_mfma_f32_16x16x32_f16 v[96:99], a[44:47], a[64:67], v[96:99]
	v_mfma_f32_16x16x32_f16 v[92:95], a[96:99], a[32:35], v[92:95]
	v_mfma_f32_16x16x32_f16 v[88:91], a[88:91], a[32:35], v[88:91]
	v_mfma_f32_16x16x32_f16 v[84:87], a[56:59], a[32:35], v[84:87]
	v_mfma_f32_16x16x32_f16 v[80:83], a[80:83], a[32:35], v[80:83]
	v_mfma_f32_16x16x32_f16 v[76:79], a[52:55], a[32:35], v[76:79]
	v_mfma_f32_16x16x32_f16 v[72:75], a[112:115], a[32:35], v[72:75]
	v_mfma_f32_16x16x32_f16 v[68:71], a[28:31], a[32:35], v[68:71]
	v_mfma_f32_16x16x32_f16 v[64:67], a[44:47], a[32:35], v[64:67]
	v_mfma_f32_16x16x32_f16 v[60:63], a[100:103], a[4:7], v[60:63]
	v_mfma_f32_16x16x32_f16 v[56:59], a[92:95], a[4:7], v[56:59]
	v_mfma_f32_16x16x32_f16 v[52:55], a[12:15], a[4:7], v[52:55]
	v_mfma_f32_16x16x32_f16 v[48:51], a[84:87], a[4:7], v[48:51]
	v_mfma_f32_16x16x32_f16 v[44:47], a[16:19], a[4:7], v[44:47]
	v_mfma_f32_16x16x32_f16 v[40:43], a[116:119], a[4:7], v[40:43]
	v_mfma_f32_16x16x32_f16 v[36:39], a[24:27], a[4:7], v[36:39]
	v_mfma_f32_16x16x32_f16 v[32:35], a[8:11], a[4:7], v[32:35]
	v_mfma_f32_16x16x32_f16 v[28:31], a[96:99], a[48:51], v[28:31]
	v_mfma_f32_16x16x32_f16 v[24:27], a[88:91], a[48:51], v[24:27]
	v_mfma_f32_16x16x32_f16 v[20:23], a[56:59], a[48:51], v[20:23]
	v_mfma_f32_16x16x32_f16 v[16:19], a[80:83], a[48:51], v[16:19]
	v_mfma_f32_16x16x32_f16 v[12:15], a[52:55], a[48:51], v[12:15]
	v_mfma_f32_16x16x32_f16 v[8:11], a[112:115], a[48:51], v[8:11]
	v_mfma_f32_16x16x32_f16 a[148:151], a[28:31], a[48:51], a[148:151]
	v_mfma_f32_16x16x32_f16 a[4:7], a[44:47], a[48:51], a[192:195]
	v_mfma_f32_16x16x32_f16 v[248:251], a[100:103], a[124:127], v[248:251]
	v_mfma_f32_16x16x32_f16 v[252:255], a[92:95], a[124:127], v[252:255]
	v_mfma_f32_16x16x32_f16 v[244:247], a[12:15], a[124:127], v[244:247]
	v_mfma_f32_16x16x32_f16 v[240:243], a[84:87], a[124:127], v[240:243]
	v_mfma_f32_16x16x32_f16 a[166:169], a[16:19], a[124:127], a[166:169]
	v_mfma_f32_16x16x32_f16 v[232:235], a[116:119], a[124:127], v[232:235]
	v_mfma_f32_16x16x32_f16 v[228:231], a[24:27], a[124:127], v[228:231]
	v_mfma_f32_16x16x32_f16 v[188:191], a[100:103], a[108:111], v[188:191]
	v_mfma_f32_16x16x32_f16 v[184:187], a[92:95], a[108:111], v[184:187]
	v_mfma_f32_16x16x32_f16 v[180:183], a[12:15], a[108:111], v[180:183]
	v_mfma_f32_16x16x32_f16 v[176:179], a[84:87], a[108:111], v[176:179]
	v_mfma_f32_16x16x32_f16 v[172:175], a[16:19], a[108:111], v[172:175]
	v_mfma_f32_16x16x32_f16 v[168:171], a[116:119], a[108:111], v[168:171]
	v_mfma_f32_16x16x32_f16 v[164:167], a[24:27], a[108:111], v[164:167]
	v_mfma_f32_16x16x32_f16 v[160:163], a[8:11], a[108:111], v[160:163]
	v_mfma_f32_16x16x32_f16 v[156:159], a[100:103], a[76:79], v[156:159]
	v_mfma_f32_16x16x32_f16 v[152:155], a[92:95], a[76:79], v[152:155]
	v_mfma_f32_16x16x32_f16 v[144:147], a[84:87], a[76:79], v[144:147]
	v_mfma_f32_16x16x32_f16 v[140:143], a[16:19], a[76:79], v[140:143]
	v_mfma_f32_16x16x32_f16 v[136:139], a[116:119], a[76:79], v[136:139]
	v_mfma_f32_16x16x32_f16 v[132:135], a[24:27], a[76:79], v[132:135]
	v_mfma_f32_16x16x32_f16 v[124:127], a[100:103], a[68:71], v[124:127]
	v_mfma_f32_16x16x32_f16 v[120:123], a[92:95], a[68:71], v[120:123]
	v_mfma_f32_16x16x32_f16 v[116:119], a[12:15], a[68:71], v[116:119]
	v_mfma_f32_16x16x32_f16 v[112:115], a[84:87], a[68:71], v[112:115]
	v_mfma_f32_16x16x32_f16 a[180:183], a[16:19], a[68:71], a[36:39]
	v_mfma_f32_16x16x32_f16 v[104:107], a[116:119], a[68:71], v[104:107]
	v_mfma_f32_16x16x32_f16 v[100:103], a[24:27], a[68:71], v[100:103]
	v_mfma_f32_16x16x32_f16 v[96:99], a[8:11], a[68:71], v[96:99]
	v_mfma_f32_16x16x32_f16 v[92:95], a[100:103], a[20:23], v[92:95]
	v_mfma_f32_16x16x32_f16 v[88:91], a[92:95], a[20:23], v[88:91]
	v_mfma_f32_16x16x32_f16 v[84:87], a[12:15], a[20:23], v[84:87]
	v_mfma_f32_16x16x32_f16 v[80:83], a[84:87], a[20:23], v[80:83]
	v_mfma_f32_16x16x32_f16 v[76:79], a[16:19], a[20:23], v[76:79]
	v_mfma_f32_16x16x32_f16 v[72:75], a[116:119], a[20:23], v[72:75]
	v_mfma_f32_16x16x32_f16 v[68:71], a[24:27], a[20:23], v[68:71]
	v_mfma_f32_16x16x32_f16 v[64:67], a[8:11], a[20:23], v[64:67]
	v_mfma_f32_16x16x32_f16 v[28:31], a[100:103], a[0:3], v[28:31]
	v_mfma_f32_16x16x32_f16 v[24:27], a[92:95], a[0:3], v[24:27]
	v_mfma_f32_16x16x32_f16 v[20:23], a[12:15], a[0:3], v[20:23]
	v_mfma_f32_16x16x32_f16 v[16:19], a[84:87], a[0:3], v[16:19]
	v_mfma_f32_16x16x32_f16 v[12:15], a[16:19], a[0:3], v[12:15]
	v_mfma_f32_16x16x32_f16 v[8:11], a[116:119], a[0:3], v[8:11]
	v_mfma_f32_16x16x32_f16 a[148:151], a[24:27], a[0:3], a[148:151]
	v_mfma_f32_16x16x32_f16 a[192:195], a[8:11], a[0:3], a[4:7]
	ds_read_b128 a[120:123], v4
	ds_read_b128 a[124:127], v4 offset:64
	ds_read_b128 a[60:63], v4 offset:256
	ds_read_b128 a[36:39], v4 offset:320
	ds_read_b128 a[104:107], v4 offset:512
	ds_read_b128 a[108:111], v4 offset:576
	ds_read_b128 a[72:75], v4 offset:768
	ds_read_b128 a[76:79], v4 offset:832
	ds_read_b128 a[64:67], v4 offset:16896
	ds_read_b128 a[68:71], v4 offset:16960
	ds_read_b128 a[32:35], v4 offset:17152
	ds_read_b128 a[20:23], v4 offset:17216
	ds_read_b128 a[40:43], v4 offset:17408
	ds_read_b128 a[4:7], v4 offset:17472
	ds_read_b128 a[48:51], v4 offset:17664
	ds_read_b128 a[0:3], v4 offset:17728
	v_accvgpr_read_b32 v4, a133
	v_add_u32_e32 v4, s10, v4
	v_add_u32_e32 v4, 0x107e0, v4
	ds_read_b128 a[96:99], v4
	ds_read_b128 a[100:103], v4 offset:64
	ds_read_b128 a[88:91], v4 offset:256
	ds_read_b128 a[92:95], v4 offset:320
	ds_read_b128 a[56:59], v4 offset:512
	ds_read_b128 a[12:15], v4 offset:576
	ds_read_b128 a[80:83], v4 offset:768
	ds_read_b128 a[84:87], v4 offset:832
	ds_read_b128 a[52:55], v4 offset:16896
	ds_read_b128 a[16:19], v4 offset:16960
	ds_read_b128 a[112:115], v4 offset:17152
	ds_read_b128 a[116:119], v4 offset:17216
	ds_read_b128 a[28:31], v4 offset:17408
	ds_read_b128 a[24:27], v4 offset:17472
	ds_read_b128 a[44:47], v4 offset:17664
	ds_read_b128 a[8:11], v4 offset:17728
	s_cbranch_scc1 .LBB0_3
; %bb.4:                                ; %Flow
	v_accvgpr_read_b32 v4, a148
	v_accvgpr_mov_b32 a133, a159
	s_mov_b32 s3, s2
	v_accvgpr_read_b32 v5, a149
	v_accvgpr_read_b32 v6, a150
	v_accvgpr_read_b32 v7, a151
	s_branch .LBB0_7
.LBB0_5:
	v_accvgpr_write_b32 a175, 0
	v_accvgpr_write_b32 a174, 0
	v_accvgpr_write_b32 a173, 0
	v_accvgpr_write_b32 a172, 0
	v_mov_b32_e32 v7, 0
	v_mov_b32_e32 v6, 0
	v_mov_b32_e32 v5, 0
	v_mov_b32_e32 v4, 0
	v_mov_b32_e32 v11, 0
	v_mov_b32_e32 v10, 0
	v_mov_b32_e32 v9, 0
	v_mov_b32_e32 v8, 0
	v_mov_b32_e32 v15, 0
	v_mov_b32_e32 v14, 0
	v_mov_b32_e32 v13, 0
	v_mov_b32_e32 v12, 0
	v_mov_b32_e32 v19, 0
	v_mov_b32_e32 v18, 0
	v_mov_b32_e32 v17, 0
	v_mov_b32_e32 v16, 0
	v_mov_b32_e32 v23, 0
	v_mov_b32_e32 v22, 0
	v_mov_b32_e32 v21, 0
	v_mov_b32_e32 v20, 0
	v_mov_b32_e32 v27, 0
	v_mov_b32_e32 v26, 0
	v_mov_b32_e32 v25, 0
	v_mov_b32_e32 v24, 0
	v_mov_b32_e32 v31, 0
	v_mov_b32_e32 v30, 0
	v_mov_b32_e32 v29, 0
	v_mov_b32_e32 v28, 0
	v_mov_b32_e32 v35, 0
	v_mov_b32_e32 v34, 0
	v_mov_b32_e32 v33, 0
	v_mov_b32_e32 v32, 0
	v_mov_b32_e32 v39, 0
	v_mov_b32_e32 v38, 0
	v_mov_b32_e32 v37, 0
	v_mov_b32_e32 v36, 0
	v_mov_b32_e32 v43, 0
	v_mov_b32_e32 v42, 0
	v_mov_b32_e32 v41, 0
	v_mov_b32_e32 v40, 0
	v_mov_b32_e32 v47, 0
	v_mov_b32_e32 v46, 0
	v_mov_b32_e32 v45, 0
	v_mov_b32_e32 v44, 0
	v_mov_b32_e32 v51, 0
	v_mov_b32_e32 v50, 0
	v_mov_b32_e32 v49, 0
	v_mov_b32_e32 v48, 0
	v_mov_b32_e32 v55, 0
	v_mov_b32_e32 v54, 0
	v_mov_b32_e32 v53, 0
	v_mov_b32_e32 v52, 0
	v_mov_b32_e32 v59, 0
	v_mov_b32_e32 v58, 0
	v_mov_b32_e32 v57, 0
	v_mov_b32_e32 v56, 0
	v_mov_b32_e32 v63, 0
	v_mov_b32_e32 v62, 0
	v_mov_b32_e32 v61, 0
	v_mov_b32_e32 v60, 0
	v_mov_b32_e32 v67, 0
	v_mov_b32_e32 v66, 0
	v_mov_b32_e32 v65, 0
	v_mov_b32_e32 v64, 0
	v_mov_b32_e32 v71, 0
	v_mov_b32_e32 v70, 0
	v_mov_b32_e32 v69, 0
	v_mov_b32_e32 v68, 0
	v_mov_b32_e32 v75, 0
	v_mov_b32_e32 v74, 0
	v_mov_b32_e32 v73, 0
	v_mov_b32_e32 v72, 0
	v_mov_b32_e32 v79, 0
	v_mov_b32_e32 v78, 0
	v_mov_b32_e32 v77, 0
	v_mov_b32_e32 v76, 0
	v_mov_b32_e32 v83, 0
	v_mov_b32_e32 v82, 0
	v_mov_b32_e32 v81, 0
	v_mov_b32_e32 v80, 0
	v_mov_b32_e32 v87, 0
	v_mov_b32_e32 v86, 0
	v_mov_b32_e32 v85, 0
	v_mov_b32_e32 v84, 0
	v_mov_b32_e32 v91, 0
	v_mov_b32_e32 v90, 0
	v_mov_b32_e32 v89, 0
	v_mov_b32_e32 v88, 0
	v_mov_b32_e32 v95, 0
	v_mov_b32_e32 v94, 0
	v_mov_b32_e32 v93, 0
	v_mov_b32_e32 v92, 0
	v_mov_b32_e32 v151, 0
	v_mov_b32_e32 v150, 0
	v_mov_b32_e32 v149, 0
	v_mov_b32_e32 v148, 0
	v_mov_b32_e32 v103, 0
	v_mov_b32_e32 v102, 0
	v_mov_b32_e32 v101, 0
	v_mov_b32_e32 v100, 0
	v_mov_b32_e32 v107, 0
	v_mov_b32_e32 v106, 0
	v_mov_b32_e32 v105, 0
	v_mov_b32_e32 v104, 0
	v_accvgpr_write_b32 a159, 0
	v_accvgpr_write_b32 a158, 0
	v_accvgpr_write_b32 a157, 0
	v_accvgpr_write_b32 a156, 0
	v_mov_b32_e32 v115, 0
	v_mov_b32_e32 v114, 0
	v_mov_b32_e32 v113, 0
	v_mov_b32_e32 v112, 0
	v_mov_b32_e32 v119, 0
	v_mov_b32_e32 v118, 0
	v_mov_b32_e32 v117, 0
	v_mov_b32_e32 v116, 0
	v_mov_b32_e32 v123, 0
	v_mov_b32_e32 v122, 0
	v_mov_b32_e32 v121, 0
	v_mov_b32_e32 v120, 0
	v_mov_b32_e32 v127, 0
	v_mov_b32_e32 v126, 0
	v_mov_b32_e32 v125, 0
	v_mov_b32_e32 v124, 0
	v_accvgpr_write_b32 a155, 0
	v_accvgpr_write_b32 a154, 0
	v_accvgpr_write_b32 a153, 0
	v_accvgpr_write_b32 a152, 0
	v_mov_b32_e32 v135, 0
	v_mov_b32_e32 v134, 0
	v_mov_b32_e32 v133, 0
	v_mov_b32_e32 v132, 0
	v_mov_b32_e32 v139, 0
	v_mov_b32_e32 v138, 0
	v_mov_b32_e32 v137, 0
	v_mov_b32_e32 v136, 0
	v_mov_b32_e32 v143, 0
	v_mov_b32_e32 v142, 0
	v_mov_b32_e32 v141, 0
	v_mov_b32_e32 v140, 0
	v_mov_b32_e32 v147, 0
	v_mov_b32_e32 v146, 0
	v_mov_b32_e32 v145, 0
	v_mov_b32_e32 v144, 0
	v_accvgpr_write_b32 a163, 0
	v_accvgpr_write_b32 a162, 0
	v_accvgpr_write_b32 a161, 0
	v_accvgpr_write_b32 a160, 0
	v_mov_b32_e32 v155, 0
	v_mov_b32_e32 v154, 0
	v_mov_b32_e32 v153, 0
	v_mov_b32_e32 v152, 0
	v_mov_b32_e32 v159, 0
	v_mov_b32_e32 v158, 0
	v_mov_b32_e32 v157, 0
	v_mov_b32_e32 v156, 0
	v_mov_b32_e32 v163, 0
	v_mov_b32_e32 v162, 0
	v_mov_b32_e32 v161, 0
	v_mov_b32_e32 v160, 0
	v_mov_b32_e32 v167, 0
	v_mov_b32_e32 v166, 0
	v_mov_b32_e32 v165, 0
	v_mov_b32_e32 v164, 0
	v_mov_b32_e32 v171, 0
	v_mov_b32_e32 v170, 0
	v_mov_b32_e32 v169, 0
	v_mov_b32_e32 v168, 0
	v_mov_b32_e32 v175, 0
	v_mov_b32_e32 v174, 0
	v_mov_b32_e32 v173, 0
	v_mov_b32_e32 v172, 0
	v_mov_b32_e32 v179, 0
	v_mov_b32_e32 v178, 0
	v_mov_b32_e32 v177, 0
	v_mov_b32_e32 v176, 0
	v_mov_b32_e32 v183, 0
	v_mov_b32_e32 v182, 0
	v_mov_b32_e32 v181, 0
	v_mov_b32_e32 v180, 0
	v_mov_b32_e32 v187, 0
	v_mov_b32_e32 v186, 0
	v_mov_b32_e32 v185, 0
	v_mov_b32_e32 v184, 0
	v_mov_b32_e32 v191, 0
	v_mov_b32_e32 v190, 0
	v_mov_b32_e32 v189, 0
	v_mov_b32_e32 v188, 0
	v_mov_b32_e32 v195, 0
	v_mov_b32_e32 v194, 0
	v_mov_b32_e32 v193, 0
	v_mov_b32_e32 v192, 0
	v_accvgpr_write_b32 a151, 0
	v_accvgpr_write_b32 a150, 0
	v_accvgpr_write_b32 a149, 0
	v_accvgpr_write_b32 a148, 0
	v_mov_b32_e32 v203, 0
	v_mov_b32_e32 v202, 0
	v_mov_b32_e32 v201, 0
	v_mov_b32_e32 v200, 0
	v_mov_b32_e32 v207, 0
	v_mov_b32_e32 v206, 0
	v_mov_b32_e32 v205, 0
	v_mov_b32_e32 v204, 0
	v_mov_b32_e32 v211, 0
	v_mov_b32_e32 v210, 0
	v_mov_b32_e32 v209, 0
	v_mov_b32_e32 v208, 0
	v_mov_b32_e32 v215, 0
	v_mov_b32_e32 v214, 0
	v_mov_b32_e32 v213, 0
	v_mov_b32_e32 v212, 0
	v_mov_b32_e32 v219, 0
	v_mov_b32_e32 v218, 0
	v_mov_b32_e32 v217, 0
	v_mov_b32_e32 v216, 0
	v_mov_b32_e32 v223, 0
	v_mov_b32_e32 v222, 0
	v_mov_b32_e32 v221, 0
	v_mov_b32_e32 v220, 0
	v_accvgpr_write_b32 a171, 0
	v_accvgpr_write_b32 a170, 0
	v_accvgpr_write_b32 a169, 0
	v_accvgpr_write_b32 a168, 0
	v_mov_b32_e32 v231, 0
	v_mov_b32_e32 v230, 0
	v_mov_b32_e32 v229, 0
	v_mov_b32_e32 v228, 0
	v_mov_b32_e32 v235, 0
	v_mov_b32_e32 v234, 0
	v_mov_b32_e32 v233, 0
	v_mov_b32_e32 v232, 0
	v_accvgpr_write_b32 a167, 0
	v_accvgpr_write_b32 a166, 0
	v_accvgpr_write_b32 a165, 0
	v_accvgpr_write_b32 a164, 0
	v_mov_b32_e32 v243, 0
	v_mov_b32_e32 v242, 0
	v_mov_b32_e32 v241, 0
	v_mov_b32_e32 v240, 0
	v_mov_b32_e32 v247, 0
	v_mov_b32_e32 v246, 0
	v_mov_b32_e32 v245, 0
	v_mov_b32_e32 v244, 0
	v_mov_b32_e32 v255, 0
	v_mov_b32_e32 v254, 0
	v_mov_b32_e32 v253, 0
	v_mov_b32_e32 v252, 0
	v_mov_b32_e32 v251, 0
	v_mov_b32_e32 v250, 0
	v_mov_b32_e32 v249, 0
	v_mov_b32_e32 v248, 0
	s_branch .LBB0_8
.LBB0_6:
	v_accvgpr_write_b32 a195, 0
	v_accvgpr_write_b32 a133, v4
	v_accvgpr_mov_b32 a194, a195
	v_accvgpr_mov_b32 a193, a195
	v_accvgpr_mov_b32 a192, a195
	v_accvgpr_read_b32 v7, a195
	v_accvgpr_read_b32 v6, a195
	v_accvgpr_read_b32 v5, a195
	v_accvgpr_read_b32 v4, a195
	v_accvgpr_read_b32 v11, a195
	v_accvgpr_read_b32 v10, a195
	v_accvgpr_read_b32 v9, a195
	v_accvgpr_read_b32 v8, a195
	v_accvgpr_read_b32 v15, a195
	v_accvgpr_read_b32 v14, a195
	v_accvgpr_read_b32 v13, a195
	v_accvgpr_read_b32 v12, a195
	v_accvgpr_read_b32 v19, a195
	v_accvgpr_read_b32 v18, a195
	v_accvgpr_read_b32 v17, a195
	v_accvgpr_read_b32 v16, a195
	v_accvgpr_read_b32 v23, a195
	v_accvgpr_read_b32 v22, a195
	v_accvgpr_read_b32 v21, a195
	v_accvgpr_read_b32 v20, a195
	v_accvgpr_read_b32 v27, a195
	v_accvgpr_read_b32 v26, a195
	v_accvgpr_read_b32 v25, a195
	v_accvgpr_read_b32 v24, a195
	v_accvgpr_read_b32 v31, a195
	v_accvgpr_read_b32 v30, a195
	v_accvgpr_read_b32 v29, a195
	v_accvgpr_read_b32 v28, a195
	v_accvgpr_read_b32 v35, a195
	v_accvgpr_read_b32 v34, a195
	v_accvgpr_read_b32 v33, a195
	v_accvgpr_read_b32 v32, a195
	v_accvgpr_read_b32 v39, a195
	v_accvgpr_read_b32 v38, a195
	v_accvgpr_read_b32 v37, a195
	v_accvgpr_read_b32 v36, a195
	v_accvgpr_read_b32 v43, a195
	v_accvgpr_read_b32 v42, a195
	v_accvgpr_read_b32 v41, a195
	v_accvgpr_read_b32 v40, a195
	v_accvgpr_read_b32 v47, a195
	v_accvgpr_read_b32 v46, a195
	v_accvgpr_read_b32 v45, a195
	v_accvgpr_read_b32 v44, a195
	v_accvgpr_read_b32 v51, a195
	v_accvgpr_read_b32 v50, a195
	v_accvgpr_read_b32 v49, a195
	v_accvgpr_read_b32 v48, a195
	v_accvgpr_read_b32 v55, a195
	v_accvgpr_read_b32 v54, a195
	v_accvgpr_read_b32 v53, a195
	v_accvgpr_read_b32 v52, a195
	v_accvgpr_read_b32 v59, a195
	v_accvgpr_read_b32 v58, a195
	v_accvgpr_read_b32 v57, a195
	v_accvgpr_read_b32 v56, a195
	v_accvgpr_read_b32 v63, a195
	v_accvgpr_read_b32 v62, a195
	v_accvgpr_read_b32 v61, a195
	v_accvgpr_read_b32 v60, a195
	v_accvgpr_read_b32 v67, a195
	v_accvgpr_read_b32 v66, a195
	v_accvgpr_read_b32 v65, a195
	v_accvgpr_read_b32 v64, a195
	v_accvgpr_read_b32 v71, a195
	v_accvgpr_read_b32 v70, a195
	v_accvgpr_read_b32 v69, a195
	v_accvgpr_read_b32 v68, a195
	v_accvgpr_read_b32 v75, a195
	v_accvgpr_read_b32 v74, a195
	v_accvgpr_read_b32 v73, a195
	v_accvgpr_read_b32 v72, a195
	v_accvgpr_read_b32 v79, a195
	v_accvgpr_read_b32 v78, a195
	v_accvgpr_read_b32 v77, a195
	v_accvgpr_read_b32 v76, a195
	v_accvgpr_read_b32 v83, a195
	v_accvgpr_read_b32 v82, a195
	v_accvgpr_read_b32 v81, a195
	v_accvgpr_read_b32 v80, a195
	v_accvgpr_read_b32 v87, a195
	v_accvgpr_read_b32 v86, a195
	v_accvgpr_read_b32 v85, a195
	v_accvgpr_read_b32 v84, a195
	v_accvgpr_read_b32 v91, a195
	v_accvgpr_read_b32 v90, a195
	v_accvgpr_read_b32 v89, a195
	v_accvgpr_read_b32 v88, a195
	v_accvgpr_read_b32 v95, a195
	v_accvgpr_read_b32 v94, a195
	v_accvgpr_read_b32 v93, a195
	v_accvgpr_read_b32 v92, a195
	v_accvgpr_read_b32 v99, a195
	v_accvgpr_read_b32 v98, a195
	v_accvgpr_read_b32 v97, a195
	v_accvgpr_read_b32 v96, a195
	v_accvgpr_read_b32 v103, a195
	v_accvgpr_read_b32 v102, a195
	v_accvgpr_read_b32 v101, a195
	v_accvgpr_read_b32 v100, a195
	v_accvgpr_read_b32 v107, a195
	v_accvgpr_read_b32 v106, a195
	v_accvgpr_read_b32 v105, a195
	v_accvgpr_read_b32 v104, a195
	v_accvgpr_mov_b32 a183, a195
	v_accvgpr_mov_b32 a182, a195
	v_accvgpr_mov_b32 a181, a195
	v_accvgpr_mov_b32 a180, a195
	v_accvgpr_read_b32 v115, a195
	v_accvgpr_read_b32 v114, a195
	v_accvgpr_read_b32 v113, a195
	v_accvgpr_read_b32 v112, a195
	v_accvgpr_read_b32 v119, a195
	v_accvgpr_read_b32 v118, a195
	v_accvgpr_read_b32 v117, a195
	v_accvgpr_read_b32 v116, a195
	v_accvgpr_read_b32 v123, a195
	v_accvgpr_read_b32 v122, a195
	v_accvgpr_read_b32 v121, a195
	v_accvgpr_read_b32 v120, a195
	v_accvgpr_read_b32 v127, a195
	v_accvgpr_read_b32 v126, a195
	v_accvgpr_read_b32 v125, a195
	v_accvgpr_read_b32 v124, a195
	v_accvgpr_mov_b32 a179, a195
	v_accvgpr_mov_b32 a178, a195
	v_accvgpr_mov_b32 a177, a195
	v_accvgpr_mov_b32 a176, a195
	v_accvgpr_read_b32 v135, a195
	v_accvgpr_read_b32 v134, a195
	v_accvgpr_read_b32 v133, a195
	v_accvgpr_read_b32 v132, a195
	v_accvgpr_read_b32 v139, a195
	v_accvgpr_read_b32 v138, a195
	v_accvgpr_read_b32 v137, a195
	v_accvgpr_read_b32 v136, a195
	v_accvgpr_read_b32 v143, a195
	v_accvgpr_read_b32 v142, a195
	v_accvgpr_read_b32 v141, a195
	v_accvgpr_read_b32 v140, a195
	v_accvgpr_read_b32 v147, a195
	v_accvgpr_read_b32 v146, a195
	v_accvgpr_read_b32 v145, a195
	v_accvgpr_read_b32 v144, a195
	v_accvgpr_mov_b32 a187, a195
	v_accvgpr_mov_b32 a186, a195
	v_accvgpr_mov_b32 a185, a195
	v_accvgpr_mov_b32 a184, a195
	v_accvgpr_read_b32 v155, a195
	v_accvgpr_read_b32 v154, a195
	v_accvgpr_read_b32 v153, a195
	v_accvgpr_read_b32 v152, a195
	v_accvgpr_read_b32 v159, a195
	v_accvgpr_read_b32 v158, a195
	v_accvgpr_read_b32 v157, a195
	v_accvgpr_read_b32 v156, a195
	v_accvgpr_read_b32 v163, a195
	v_accvgpr_read_b32 v162, a195
	v_accvgpr_read_b32 v161, a195
	v_accvgpr_read_b32 v160, a195
	v_accvgpr_read_b32 v167, a195
	v_accvgpr_read_b32 v166, a195
	v_accvgpr_read_b32 v165, a195
	v_accvgpr_read_b32 v164, a195
	v_accvgpr_read_b32 v171, a195
	v_accvgpr_read_b32 v170, a195
	v_accvgpr_read_b32 v169, a195
	v_accvgpr_read_b32 v168, a195
	v_accvgpr_read_b32 v175, a195
	v_accvgpr_read_b32 v174, a195
	v_accvgpr_read_b32 v173, a195
	v_accvgpr_read_b32 v172, a195
	v_accvgpr_read_b32 v179, a195
	v_accvgpr_read_b32 v178, a195
	v_accvgpr_read_b32 v177, a195
	v_accvgpr_read_b32 v176, a195
	v_accvgpr_read_b32 v183, a195
	v_accvgpr_read_b32 v182, a195
	v_accvgpr_read_b32 v181, a195
	v_accvgpr_read_b32 v180, a195
	v_accvgpr_read_b32 v187, a195
	v_accvgpr_read_b32 v186, a195
	v_accvgpr_read_b32 v185, a195
	v_accvgpr_read_b32 v184, a195
	v_accvgpr_read_b32 v191, a195
	v_accvgpr_read_b32 v190, a195
	v_accvgpr_read_b32 v189, a195
	v_accvgpr_read_b32 v188, a195
	v_accvgpr_read_b32 v195, a195
	v_accvgpr_read_b32 v194, a195
	v_accvgpr_read_b32 v193, a195
	v_accvgpr_read_b32 v192, a195
	v_accvgpr_mov_b32 a175, a195
	v_accvgpr_mov_b32 a174, a195
	v_accvgpr_mov_b32 a173, a195
	v_accvgpr_mov_b32 a172, a195
	v_accvgpr_read_b32 v203, a195
	v_accvgpr_read_b32 v202, a195
	v_accvgpr_read_b32 v201, a195
	v_accvgpr_read_b32 v200, a195
	v_accvgpr_read_b32 v207, a195
	v_accvgpr_read_b32 v206, a195
	v_accvgpr_read_b32 v205, a195
	v_accvgpr_read_b32 v204, a195
	v_accvgpr_read_b32 v211, a195
	v_accvgpr_read_b32 v210, a195
	v_accvgpr_read_b32 v209, a195
	v_accvgpr_read_b32 v208, a195
	v_accvgpr_read_b32 v215, a195
	v_accvgpr_read_b32 v214, a195
	v_accvgpr_read_b32 v213, a195
	v_accvgpr_read_b32 v212, a195
	v_accvgpr_read_b32 v219, a195
	v_accvgpr_read_b32 v218, a195
	v_accvgpr_read_b32 v217, a195
	v_accvgpr_read_b32 v216, a195
	v_accvgpr_read_b32 v223, a195
	v_accvgpr_read_b32 v222, a195
	v_accvgpr_read_b32 v221, a195
	v_accvgpr_read_b32 v220, a195
	v_accvgpr_mov_b32 a191, a195
	v_accvgpr_mov_b32 a190, a195
	v_accvgpr_mov_b32 a189, a195
	v_accvgpr_mov_b32 a188, a195
	v_accvgpr_read_b32 v231, a195
	v_accvgpr_read_b32 v230, a195
	v_accvgpr_read_b32 v229, a195
	v_accvgpr_read_b32 v228, a195
	v_accvgpr_read_b32 v235, a195
	v_accvgpr_read_b32 v234, a195
	v_accvgpr_read_b32 v233, a195
	v_accvgpr_read_b32 v232, a195
	v_accvgpr_mov_b32 a169, a195
	v_accvgpr_mov_b32 a168, a195
	v_accvgpr_mov_b32 a167, a195
	v_accvgpr_mov_b32 a166, a195
	v_accvgpr_read_b32 v243, a195
	v_accvgpr_read_b32 v242, a195
	v_accvgpr_read_b32 v241, a195
	v_accvgpr_read_b32 v240, a195
	v_accvgpr_read_b32 v247, a195
	v_accvgpr_read_b32 v246, a195
	v_accvgpr_read_b32 v245, a195
	v_accvgpr_read_b32 v244, a195
	v_accvgpr_read_b32 v255, a195
	v_accvgpr_read_b32 v254, a195
	v_accvgpr_read_b32 v253, a195
	v_accvgpr_read_b32 v252, a195
	v_accvgpr_read_b32 v251, a195
	v_accvgpr_read_b32 v250, a195
	v_accvgpr_read_b32 v249, a195
	v_accvgpr_read_b32 v248, a195
.LBB0_7:                                ; %._crit_edge.loopexit.peel.begin
	s_and_b32 s16, s3, 1
	s_cmp_eq_u32 s3, s2
	s_mul_i32 s2, s16, 0x8400
	s_cselect_b64 vcc, -1, 0
	s_add_i32 s17, s2, 0
	s_add_i32 s17, s17, s21
	v_bfrev_b32_e32 v0, 1
	v_accvgpr_read_b32 v1, a140
	s_and_b32 s9, s9, 0xffff
	s_mov_b32 s11, 0x27000
	s_mov_b32 s10, 0x7ffffffe
	v_cndmask_b32_e32 v1, v1, v0, vcc
	s_mov_b32 m0, s17
	; wait_asyncmark(0)
	s_waitcnt vmcnt(0) lgkmcnt(0)
	s_barrier
	buffer_load_dwordx4 v1, s[8:11], 0 offen lds
	v_accvgpr_read_b32 v1, a141
	s_add_i32 m0, s17, 0x1080
	v_cndmask_b32_e32 v1, v1, v0, vcc
	buffer_load_dwordx4 v1, s[8:11], 0 offen lds
	v_accvgpr_read_b32 v1, a142
	s_add_i32 m0, s17, 0x2100
	v_cndmask_b32_e32 v1, v1, v0, vcc
	buffer_load_dwordx4 v1, s[8:11], 0 offen lds
	v_accvgpr_read_b32 v1, a143
	s_add_i32 m0, s17, 0x3180
	v_cndmask_b32_e32 v1, v1, v0, vcc
	buffer_load_dwordx4 v1, s[8:11], 0 offen lds
	v_accvgpr_read_b32 v1, a144
	s_add_i32 m0, s17, 0x4200
	v_cndmask_b32_e32 v1, v1, v0, vcc
	buffer_load_dwordx4 v1, s[8:11], 0 offen lds
	v_accvgpr_read_b32 v1, a145
	s_add_i32 m0, s17, 0x5280
	v_cndmask_b32_e32 v1, v1, v0, vcc
	buffer_load_dwordx4 v1, s[8:11], 0 offen lds
	v_accvgpr_read_b32 v1, a146
	s_add_i32 m0, s17, 0x6300
	v_cndmask_b32_e32 v1, v1, v0, vcc
	buffer_load_dwordx4 v1, s[8:11], 0 offen lds
	v_accvgpr_read_b32 v1, a147
	s_add_i32 m0, s17, 0x7380
	v_cndmask_b32_e32 v1, v1, v0, vcc
	buffer_load_dwordx4 v1, s[8:11], 0 offen lds
	v_accvgpr_read_b32 v1, a131
	s_and_b32 s1, s1, 0xffff
	s_mov_b32 s2, s10
	s_mov_b32 s3, s11
	s_add_i32 m0, s17, 0x107e0
	v_cndmask_b32_e32 v1, v1, v0, vcc
	buffer_load_dwordx4 v1, s[0:3], 0 offen lds
	v_accvgpr_read_b32 v1, a132
	s_add_i32 m0, s17, 0x11860
	v_cndmask_b32_e32 v1, v1, v0, vcc
	buffer_load_dwordx4 v1, s[0:3], 0 offen lds
	v_accvgpr_read_b32 v1, a133
	s_add_i32 m0, s17, 0x128e0
	v_cndmask_b32_e32 v1, v1, v0, vcc
	buffer_load_dwordx4 v1, s[0:3], 0 offen lds
	v_accvgpr_read_b32 v1, a134
	s_add_i32 m0, s17, 0x13960
	v_cndmask_b32_e32 v1, v1, v0, vcc
	buffer_load_dwordx4 v1, s[0:3], 0 offen lds
	v_accvgpr_read_b32 v1, a135
	s_add_i32 m0, s17, 0x149e0
	v_cndmask_b32_e32 v1, v1, v0, vcc
	buffer_load_dwordx4 v1, s[0:3], 0 offen lds
	v_accvgpr_read_b32 v1, a136
	s_add_i32 m0, s17, 0x15a60
	v_cndmask_b32_e32 v1, v1, v0, vcc
	buffer_load_dwordx4 v1, s[0:3], 0 offen lds
	v_accvgpr_read_b32 v1, a137
	s_add_i32 m0, s17, 0x16ae0
	v_cndmask_b32_e32 v1, v1, v0, vcc
	buffer_load_dwordx4 v1, s[0:3], 0 offen lds
	v_accvgpr_read_b32 v1, a138
	s_add_i32 m0, s17, 0x17b60
	v_cndmask_b32_e32 v0, v1, v0, vcc
	buffer_load_dwordx4 v0, s[0:3], 0 offen lds
	v_mfma_f32_16x16x32_f16 v[60:63], a[96:99], a[40:43], v[60:63]
	s_lshl_b32 s0, s16, 14
	s_xor_b32 s0, s0, 0x4000
	s_lshr_b32 s1, s0, 5
	v_mfma_f32_16x16x32_f16 v[56:59], a[88:91], a[40:43], v[56:59]
	s_or_b32 s1, s1, s0
	s_lshl1_add_u32 s0, s1, 0
	v_accvgpr_read_b32 v0, a139
	v_mfma_f32_16x16x32_f16 v[52:55], a[56:59], a[40:43], v[52:55]
	v_add_u32_e32 v0, s0, v0
	; asyncmark
	v_mfma_f32_16x16x32_f16 v[48:51], a[80:83], a[40:43], v[48:51]
	s_barrier
	v_mfma_f32_16x16x32_f16 v[44:47], a[52:55], a[40:43], v[44:47]
	v_mfma_f32_16x16x32_f16 v[40:43], a[112:115], a[40:43], v[40:43]
	v_mfma_f32_16x16x32_f16 v[36:39], a[28:31], a[40:43], v[36:39]
	v_mfma_f32_16x16x32_f16 v[32:35], a[44:47], a[40:43], v[32:35]
	v_mfma_f32_16x16x32_f16 v[248:251], a[96:99], a[120:123], v[248:251]
	v_mfma_f32_16x16x32_f16 v[252:255], a[88:91], a[120:123], v[252:255]
	v_mfma_f32_16x16x32_f16 v[244:247], a[56:59], a[120:123], v[244:247]
	v_mfma_f32_16x16x32_f16 v[240:243], a[80:83], a[120:123], v[240:243]
	v_mfma_f32_16x16x32_f16 a[164:167], a[52:55], a[120:123], a[166:169]
	v_mfma_f32_16x16x32_f16 v[232:235], a[112:115], a[120:123], v[232:235]
	v_mfma_f32_16x16x32_f16 v[228:231], a[28:31], a[120:123], v[228:231]
	v_mfma_f32_16x16x32_f16 a[168:171], a[44:47], a[120:123], a[188:191]
	v_mfma_f32_16x16x32_f16 v[220:223], a[96:99], a[60:63], v[220:223]
	v_mfma_f32_16x16x32_f16 v[216:219], a[88:91], a[60:63], v[216:219]
	v_mfma_f32_16x16x32_f16 v[212:215], a[56:59], a[60:63], v[212:215]
	v_mfma_f32_16x16x32_f16 v[208:211], a[80:83], a[60:63], v[208:211]
	v_mfma_f32_16x16x32_f16 v[204:207], a[52:55], a[60:63], v[204:207]
	v_mfma_f32_16x16x32_f16 v[200:203], a[112:115], a[60:63], v[200:203]
	v_mfma_f32_16x16x32_f16 a[148:151], a[28:31], a[60:63], a[172:175]
	v_mfma_f32_16x16x32_f16 v[192:195], a[44:47], a[60:63], v[192:195]
	v_mfma_f32_16x16x32_f16 v[188:191], a[96:99], a[104:107], v[188:191]
	v_mfma_f32_16x16x32_f16 v[184:187], a[88:91], a[104:107], v[184:187]
	v_mfma_f32_16x16x32_f16 v[180:183], a[56:59], a[104:107], v[180:183]
	v_mfma_f32_16x16x32_f16 v[176:179], a[80:83], a[104:107], v[176:179]
	v_mfma_f32_16x16x32_f16 v[172:175], a[52:55], a[104:107], v[172:175]
	v_mfma_f32_16x16x32_f16 v[168:171], a[112:115], a[104:107], v[168:171]
	v_mfma_f32_16x16x32_f16 v[164:167], a[28:31], a[104:107], v[164:167]
	v_mfma_f32_16x16x32_f16 v[160:163], a[44:47], a[104:107], v[160:163]
	v_mfma_f32_16x16x32_f16 v[156:159], a[96:99], a[72:75], v[156:159]
	v_mfma_f32_16x16x32_f16 v[152:155], a[88:91], a[72:75], v[152:155]
	v_mfma_f32_16x16x32_f16 a[160:163], a[56:59], a[72:75], a[184:187]
	v_mfma_f32_16x16x32_f16 v[144:147], a[80:83], a[72:75], v[144:147]
	v_mfma_f32_16x16x32_f16 v[140:143], a[52:55], a[72:75], v[140:143]
	v_mfma_f32_16x16x32_f16 v[136:139], a[112:115], a[72:75], v[136:139]
	v_mfma_f32_16x16x32_f16 v[132:135], a[28:31], a[72:75], v[132:135]
	v_mfma_f32_16x16x32_f16 a[152:155], a[44:47], a[72:75], a[176:179]
	v_mfma_f32_16x16x32_f16 v[124:127], a[96:99], a[64:67], v[124:127]
	v_mfma_f32_16x16x32_f16 v[120:123], a[88:91], a[64:67], v[120:123]
	v_mfma_f32_16x16x32_f16 v[116:119], a[56:59], a[64:67], v[116:119]
	v_mfma_f32_16x16x32_f16 v[112:115], a[80:83], a[64:67], v[112:115]
	v_mfma_f32_16x16x32_f16 a[156:159], a[52:55], a[64:67], a[180:183]
	v_mfma_f32_16x16x32_f16 v[104:107], a[112:115], a[64:67], v[104:107]
	v_mfma_f32_16x16x32_f16 v[100:103], a[28:31], a[64:67], v[100:103]
	v_mfma_f32_16x16x32_f16 v[96:99], a[44:47], a[64:67], v[96:99]
	v_mfma_f32_16x16x32_f16 v[92:95], a[96:99], a[32:35], v[92:95]
	v_mfma_f32_16x16x32_f16 v[88:91], a[88:91], a[32:35], v[88:91]
	v_mfma_f32_16x16x32_f16 v[84:87], a[56:59], a[32:35], v[84:87]
	v_mfma_f32_16x16x32_f16 v[80:83], a[80:83], a[32:35], v[80:83]
	v_mfma_f32_16x16x32_f16 v[76:79], a[52:55], a[32:35], v[76:79]
	v_mfma_f32_16x16x32_f16 v[72:75], a[112:115], a[32:35], v[72:75]
	v_mfma_f32_16x16x32_f16 v[68:71], a[28:31], a[32:35], v[68:71]
	v_mfma_f32_16x16x32_f16 v[64:67], a[44:47], a[32:35], v[64:67]
	v_mfma_f32_16x16x32_f16 v[60:63], a[100:103], a[4:7], v[60:63]
	v_mfma_f32_16x16x32_f16 v[56:59], a[92:95], a[4:7], v[56:59]
	v_mfma_f32_16x16x32_f16 v[52:55], a[12:15], a[4:7], v[52:55]
	v_mfma_f32_16x16x32_f16 v[48:51], a[84:87], a[4:7], v[48:51]
	v_mfma_f32_16x16x32_f16 v[44:47], a[16:19], a[4:7], v[44:47]
	v_mfma_f32_16x16x32_f16 v[40:43], a[116:119], a[4:7], v[40:43]
	v_mfma_f32_16x16x32_f16 v[36:39], a[24:27], a[4:7], v[36:39]
	v_mfma_f32_16x16x32_f16 v[32:35], a[8:11], a[4:7], v[32:35]
	v_mfma_f32_16x16x32_f16 v[28:31], a[96:99], a[48:51], v[28:31]
	v_mfma_f32_16x16x32_f16 v[24:27], a[88:91], a[48:51], v[24:27]
	v_mfma_f32_16x16x32_f16 v[20:23], a[56:59], a[48:51], v[20:23]
	v_mfma_f32_16x16x32_f16 v[16:19], a[80:83], a[48:51], v[16:19]
	v_mfma_f32_16x16x32_f16 v[12:15], a[52:55], a[48:51], v[12:15]
	v_mfma_f32_16x16x32_f16 v[8:11], a[112:115], a[48:51], v[8:11]
	v_mfma_f32_16x16x32_f16 v[4:7], a[28:31], a[48:51], v[4:7]
	v_mfma_f32_16x16x32_f16 a[4:7], a[44:47], a[48:51], a[192:195]
	v_mfma_f32_16x16x32_f16 v[248:251], a[100:103], a[124:127], v[248:251]
	v_mfma_f32_16x16x32_f16 v[252:255], a[92:95], a[124:127], v[252:255]
	v_mfma_f32_16x16x32_f16 v[244:247], a[12:15], a[124:127], v[244:247]
	v_mfma_f32_16x16x32_f16 v[240:243], a[84:87], a[124:127], v[240:243]
	v_mfma_f32_16x16x32_f16 a[164:167], a[16:19], a[124:127], a[164:167]
	v_mfma_f32_16x16x32_f16 v[232:235], a[116:119], a[124:127], v[232:235]
	v_mfma_f32_16x16x32_f16 v[228:231], a[24:27], a[124:127], v[228:231]
	v_mfma_f32_16x16x32_f16 a[168:171], a[8:11], a[124:127], a[168:171]
	v_mfma_f32_16x16x32_f16 v[220:223], a[100:103], a[36:39], v[220:223]
	v_mfma_f32_16x16x32_f16 v[216:219], a[92:95], a[36:39], v[216:219]
	v_mfma_f32_16x16x32_f16 v[212:215], a[12:15], a[36:39], v[212:215]
	v_mfma_f32_16x16x32_f16 v[208:211], a[84:87], a[36:39], v[208:211]
	v_mfma_f32_16x16x32_f16 v[204:207], a[16:19], a[36:39], v[204:207]
	v_mfma_f32_16x16x32_f16 v[200:203], a[116:119], a[36:39], v[200:203]
	v_mfma_f32_16x16x32_f16 a[148:151], a[24:27], a[36:39], a[148:151]
	v_mfma_f32_16x16x32_f16 v[192:195], a[8:11], a[36:39], v[192:195]
	v_mfma_f32_16x16x32_f16 v[188:191], a[100:103], a[108:111], v[188:191]
	v_mfma_f32_16x16x32_f16 v[184:187], a[92:95], a[108:111], v[184:187]
	v_mfma_f32_16x16x32_f16 v[180:183], a[12:15], a[108:111], v[180:183]
	v_mfma_f32_16x16x32_f16 v[176:179], a[84:87], a[108:111], v[176:179]
	v_mfma_f32_16x16x32_f16 v[172:175], a[16:19], a[108:111], v[172:175]
	v_mfma_f32_16x16x32_f16 v[168:171], a[116:119], a[108:111], v[168:171]
	v_mfma_f32_16x16x32_f16 v[164:167], a[24:27], a[108:111], v[164:167]
	v_mfma_f32_16x16x32_f16 v[160:163], a[8:11], a[108:111], v[160:163]
	v_mfma_f32_16x16x32_f16 v[156:159], a[100:103], a[76:79], v[156:159]
	v_mfma_f32_16x16x32_f16 v[152:155], a[92:95], a[76:79], v[152:155]
	v_mfma_f32_16x16x32_f16 a[160:163], a[12:15], a[76:79], a[160:163]
	v_mfma_f32_16x16x32_f16 v[144:147], a[84:87], a[76:79], v[144:147]
	v_mfma_f32_16x16x32_f16 v[140:143], a[16:19], a[76:79], v[140:143]
	v_mfma_f32_16x16x32_f16 v[136:139], a[116:119], a[76:79], v[136:139]
	v_mfma_f32_16x16x32_f16 v[132:135], a[24:27], a[76:79], v[132:135]
	v_mfma_f32_16x16x32_f16 a[152:155], a[8:11], a[76:79], a[152:155]
	v_mfma_f32_16x16x32_f16 v[124:127], a[100:103], a[68:71], v[124:127]
	v_mfma_f32_16x16x32_f16 v[120:123], a[92:95], a[68:71], v[120:123]
	v_mfma_f32_16x16x32_f16 v[116:119], a[12:15], a[68:71], v[116:119]
	v_mfma_f32_16x16x32_f16 v[112:115], a[84:87], a[68:71], v[112:115]
	v_mfma_f32_16x16x32_f16 a[156:159], a[16:19], a[68:71], a[156:159]
	v_mfma_f32_16x16x32_f16 v[104:107], a[116:119], a[68:71], v[104:107]
	v_mfma_f32_16x16x32_f16 v[100:103], a[24:27], a[68:71], v[100:103]
	v_mfma_f32_16x16x32_f16 v[148:151], a[8:11], a[68:71], v[96:99]
	v_mfma_f32_16x16x32_f16 v[92:95], a[100:103], a[20:23], v[92:95]
	v_mfma_f32_16x16x32_f16 v[88:91], a[92:95], a[20:23], v[88:91]
	v_mfma_f32_16x16x32_f16 v[84:87], a[12:15], a[20:23], v[84:87]
	v_mfma_f32_16x16x32_f16 v[80:83], a[84:87], a[20:23], v[80:83]
	v_mfma_f32_16x16x32_f16 v[76:79], a[16:19], a[20:23], v[76:79]
	v_mfma_f32_16x16x32_f16 v[72:75], a[116:119], a[20:23], v[72:75]
	v_mfma_f32_16x16x32_f16 v[68:71], a[24:27], a[20:23], v[68:71]
	v_mfma_f32_16x16x32_f16 v[64:67], a[8:11], a[20:23], v[64:67]
	v_mfma_f32_16x16x32_f16 v[28:31], a[100:103], a[0:3], v[28:31]
	v_mfma_f32_16x16x32_f16 v[24:27], a[92:95], a[0:3], v[24:27]
	v_mfma_f32_16x16x32_f16 v[20:23], a[12:15], a[0:3], v[20:23]
	v_mfma_f32_16x16x32_f16 v[16:19], a[84:87], a[0:3], v[16:19]
	v_mfma_f32_16x16x32_f16 v[12:15], a[16:19], a[0:3], v[12:15]
	v_mfma_f32_16x16x32_f16 v[8:11], a[116:119], a[0:3], v[8:11]
	v_mfma_f32_16x16x32_f16 v[4:7], a[24:27], a[0:3], v[4:7]
	v_mfma_f32_16x16x32_f16 a[172:175], a[8:11], a[0:3], a[4:7]
	ds_read_b128 a[120:123], v0
	ds_read_b128 a[124:127], v0 offset:64
	ds_read_b128 a[60:63], v0 offset:256
	ds_read_b128 a[36:39], v0 offset:320
	ds_read_b128 a[104:107], v0 offset:512
	ds_read_b128 a[108:111], v0 offset:576
	ds_read_b128 a[72:75], v0 offset:768
	ds_read_b128 a[76:79], v0 offset:832
	ds_read_b128 a[64:67], v0 offset:16896
	ds_read_b128 a[68:71], v0 offset:16960
	ds_read_b128 a[32:35], v0 offset:17152
	ds_read_b128 a[20:23], v0 offset:17216
	ds_read_b128 a[40:43], v0 offset:17408
	ds_read_b128 a[4:7], v0 offset:17472
	ds_read_b128 a[48:51], v0 offset:17664
	ds_read_b128 a[0:3], v0 offset:17728
	v_accvgpr_read_b32 v0, a130
	v_add_u32_e32 v0, s0, v0
	v_add_u32_e32 v0, 0x107e0, v0
	ds_read_b128 a[96:99], v0
	ds_read_b128 a[100:103], v0 offset:64
	ds_read_b128 a[88:91], v0 offset:256
	ds_read_b128 a[92:95], v0 offset:320
	ds_read_b128 a[56:59], v0 offset:512
	ds_read_b128 a[12:15], v0 offset:576
	ds_read_b128 a[80:83], v0 offset:768
	ds_read_b128 a[84:87], v0 offset:832
	ds_read_b128 a[52:55], v0 offset:16896
	ds_read_b128 a[16:19], v0 offset:16960
	ds_read_b128 a[112:115], v0 offset:17152
	ds_read_b128 a[116:119], v0 offset:17216
	ds_read_b128 a[28:31], v0 offset:17408
	ds_read_b128 a[24:27], v0 offset:17472
	ds_read_b128 a[44:47], v0 offset:17664
	ds_read_b128 a[8:11], v0 offset:17728
.LBB0_8:                                ; %Flow388
	s_waitcnt lgkmcnt(13)
	v_mfma_f32_16x16x32_f16 v[184:187], a[88:91], a[104:107], v[184:187]
	s_load_dword s0, s[4:5], 0x2c
	s_lshr_b32 s1, s20, 6
	s_lshl_b32 s1, s1, 3
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_16x16x32_f16 v[196:199], a[92:95], a[108:111], v[184:187]
	s_mov_b32 s39, 0x27000
	s_mul_i32 s2, s7, s0
	s_ashr_i32 s3, s2, 31
	v_mfma_f32_16x16x32_f16 v[184:187], a[96:99], a[60:63], v[220:223]
	s_lshl_b64 s[2:3], s[2:3], 1
	s_mov_b32 s38, 0x7ffffffe
	v_mfma_f32_16x16x32_f16 v[220:223], a[100:103], a[36:39], v[184:187]
	v_mfma_f32_16x16x32_f16 v[184:187], a[96:99], a[104:107], v[188:191]
	v_mfma_f32_16x16x32_f16 v[0:3], a[96:99], a[120:123], v[248:251]
	s_nop 1
	v_accvgpr_read_b32 v191, a151
	v_accvgpr_read_b32 v190, a150
	v_accvgpr_read_b32 v189, a149
	v_mfma_f32_16x16x32_f16 v[116:119], a[56:59], a[64:67], v[116:119]
	v_accvgpr_read_b32 v188, a148
	v_mfma_f32_16x16x32_f16 v[128:131], a[100:103], a[108:111], v[184:187]
	s_nop 2
	v_accvgpr_read_b32 v187, a167
	v_accvgpr_read_b32 v186, a166
	v_accvgpr_read_b32 v185, a165
	v_accvgpr_read_b32 v184, a164
	v_mfma_f32_16x16x32_f16 v[248:251], a[100:103], a[124:127], v[0:3]
	v_mfma_f32_16x16x32_f16 v[0:3], a[56:59], a[32:35], v[84:87]
	v_mfma_f32_16x16x32_f16 v[244:247], a[56:59], a[120:123], v[244:247]
	v_mfma_f32_16x16x32_f16 v[240:243], a[80:83], a[120:123], v[240:243]
	v_mfma_f32_16x16x32_f16 v[232:235], a[112:115], a[120:123], v[232:235]
	v_mfma_f32_16x16x32_f16 v[156:159], a[96:99], a[72:75], v[156:159]
	v_mfma_f32_16x16x32_f16 v[224:227], a[12:15], a[68:71], v[116:119]
	v_mfma_f32_16x16x32_f16 v[116:119], a[56:59], a[60:63], v[212:215]
	v_mfma_f32_16x16x32_f16 v[132:135], a[28:31], a[72:75], v[132:135]
	v_mfma_f32_16x16x32_f16 v[184:187], a[52:55], a[120:123], v[184:187]
	v_mfma_f32_16x16x32_f16 v[84:87], a[12:15], a[20:23], v[0:3]
	v_mfma_f32_16x16x32_f16 v[244:247], a[12:15], a[124:127], v[244:247]
	v_mfma_f32_16x16x32_f16 v[240:243], a[84:87], a[124:127], v[240:243]
	s_nop 5
	v_cvt_pk_f16_f32 v84, v84, v85
	v_cvt_pk_f16_f32 v85, v86, v87
	v_mfma_f32_16x16x32_f16 v[232:235], a[116:119], a[124:127], v[232:235]
	v_mfma_f32_16x16x32_f16 v[96:99], a[100:103], a[76:79], v[156:159]
	v_mfma_f32_16x16x32_f16 v[124:127], a[96:99], a[64:67], v[124:127]
	v_mfma_f32_16x16x32_f16 v[156:159], a[28:31], a[104:107], v[164:167]
	v_mfma_f32_16x16x32_f16 v[116:119], a[12:15], a[36:39], v[116:119]
	v_mfma_f32_16x16x32_f16 v[112:115], a[80:83], a[64:67], v[112:115]
	v_mfma_f32_16x16x32_f16 v[140:143], a[52:55], a[72:75], v[140:143]
	v_mfma_f32_16x16x32_f16 v[0:3], a[24:27], a[76:79], v[132:135]
	v_mfma_f32_16x16x32_f16 v[184:187], a[16:19], a[124:127], v[184:187]
	s_nop 1
	v_accvgpr_read_b32 v132, a168
	v_accvgpr_read_b32 v133, a169
	v_accvgpr_read_b32 v134, a170
	v_mfma_f32_16x16x32_f16 v[64:67], a[44:47], a[32:35], v[64:67]
	v_accvgpr_read_b32 v135, a171
	v_mfma_f32_16x16x32_f16 v[108:111], a[100:103], a[68:71], v[124:127]
	v_mfma_f32_16x16x32_f16 v[124:127], a[28:31], a[120:123], v[228:231]
	v_mfma_f32_16x16x32_f16 v[228:231], a[24:27], a[108:111], v[156:159]
	v_mfma_f32_16x16x32_f16 v[156:159], a[88:91], a[120:123], v[252:255]
	v_mfma_f32_16x16x32_f16 v[252:255], a[84:87], a[68:71], v[112:115]
	v_mfma_f32_16x16x32_f16 v[112:115], a[52:55], a[60:63], v[204:207]
	v_mfma_f32_16x16x32_f16 v[204:207], a[16:19], a[76:79], v[140:143]
	v_mfma_f32_16x16x32_f16 v[140:143], a[44:47], a[60:63], v[192:195]
	v_mfma_f32_16x16x32_f16 v[44:47], a[52:55], a[40:43], v[44:47]
	s_nop 1
	v_cvt_pk_f16_f32 v194, v248, v249
	v_cvt_pk_f16_f32 v248, v244, v245
	v_cvt_pk_f16_f32 v245, v242, v243
	v_cvt_pk_f16_f32 v242, v184, v185
	v_cvt_pk_f16_f32 v184, v232, v233
	v_cvt_pk_f16_f32 v232, v220, v221
	v_cvt_pk_f16_f32 v221, v118, v119
	v_cvt_pk_f16_f32 v118, v0, v1
	v_cvt_pk_f16_f32 v119, v2, v3
	v_mfma_f32_16x16x32_f16 v[0:3], a[8:11], a[20:23], v[64:67]
	v_cvt_pk_f16_f32 v195, v250, v251
	v_cvt_pk_f16_f32 v249, v246, v247
	v_cvt_pk_f16_f32 v244, v240, v241
	v_mfma_f32_16x16x32_f16 v[32:35], a[44:47], a[40:43], v[32:35]
	v_cvt_pk_f16_f32 v243, v186, v187
	v_cvt_pk_f16_f32 v185, v234, v235
	v_cvt_pk_f16_f32 v233, v222, v223
	v_mfma_f32_16x16x32_f16 v[20:23], a[56:59], a[48:51], v[20:23]
	v_cvt_pk_f16_f32 v64, v0, v1
	v_cvt_pk_f16_f32 v65, v2, v3
	v_cvt_pk_f16_f32 v220, v116, v117
	v_mfma_f32_16x16x32_f16 v[0:3], a[16:19], a[4:7], v[44:47]
	v_mfma_f32_16x16x32_f16 a[130:133], a[44:47], a[48:51], a[172:175]
	v_mfma_f32_16x16x32_f16 v[16:19], a[80:83], a[48:51], v[16:19]
	s_nop 5
	v_cvt_pk_f16_f32 v44, v0, v1
	v_cvt_pk_f16_f32 v45, v2, v3
	v_mfma_f32_16x16x32_f16 v[0:3], a[8:11], a[4:7], v[32:35]
	v_mfma_f32_16x16x32_f16 v[24:27], a[88:91], a[48:51], v[24:27]
	v_mfma_f32_16x16x32_f16 v[8:11], a[112:115], a[48:51], v[8:11]
	s_nop 5
	v_cvt_pk_f16_f32 v32, v0, v1
	v_cvt_pk_f16_f32 v33, v2, v3
	v_mfma_f32_16x16x32_f16 v[0:3], a[12:15], a[0:3], v[20:23]
	v_mfma_f32_16x16x32_f16 v[28:31], a[96:99], a[48:51], v[28:31]
	v_mfma_f32_16x16x32_f16 v[12:15], a[52:55], a[48:51], v[12:15]
	s_nop 5
	v_cvt_pk_f16_f32 v20, v0, v1
	v_cvt_pk_f16_f32 v21, v2, v3
	v_accvgpr_read_b32 v0, a130
	v_accvgpr_read_b32 v1, a131
	v_accvgpr_read_b32 v2, a132
	v_accvgpr_read_b32 v3, a133
	v_mfma_f32_16x16x32_f16 v[4:7], a[28:31], a[48:51], v[4:7]
	s_nop 0
	v_mfma_f32_16x16x32_f16 v[0:3], a[8:11], a[0:3], v[0:3]
	v_mfma_f32_16x16x32_f16 v[40:43], a[112:115], a[40:43], v[40:43]
	v_mfma_f32_16x16x32_f16 v[36:39], a[28:31], a[40:43], v[36:39]
	s_nop 5
	v_cvt_pk_f16_f32 v0, v0, v1
	v_cvt_pk_f16_f32 v1, v2, v3
	v_accvgpr_read_b32 v2, a129
	v_mfma_f32_16x16x32_f16 v[16:19], a[84:87], a[0:3], v[16:19]
	v_and_or_b32 v2, s1, 16, v2
	s_add_u32 s1, s12, s2
	s_addc_u32 s4, s13, s3
	v_mfma_f32_16x16x32_f16 v[24:27], a[92:95], a[0:3], v[24:27]
	s_ashr_i32 s7, s6, 31
	s_lshl_b64 s[2:3], s[6:7], 1
	v_or_b32_e32 v3, 32, v2
	v_mfma_f32_16x16x32_f16 v[8:11], a[116:119], a[0:3], v[8:11]
	v_cvt_pk_f16_f32 v16, v16, v17
	v_cvt_pk_f16_f32 v17, v18, v19
	v_accvgpr_read_b32 v18, a128
	v_mfma_f32_16x16x32_f16 v[28:31], a[100:103], a[0:3], v[28:31]
	v_lshrrev_b32_e32 v18, 2, v18
	v_and_b32_e32 v18, 28, v18
	v_cvt_pk_f16_f32 v24, v24, v25
	v_mfma_f32_16x16x32_f16 v[12:15], a[16:19], a[0:3], v[12:15]
	v_cvt_pk_f16_f32 v25, v26, v27
	v_cvt_pk_f16_f32 v8, v8, v9
	v_cvt_pk_f16_f32 v9, v10, v11
	v_mfma_f32_16x16x32_f16 v[4:7], a[24:27], a[0:3], v[4:7]
	v_cvt_pk_f16_f32 v28, v28, v29
	v_cvt_pk_f16_f32 v29, v30, v31
	v_or_b32_e32 v10, 0x80, v2
	v_mfma_f32_16x16x32_f16 v[40:43], a[116:119], a[4:7], v[40:43]
	v_cvt_pk_f16_f32 v12, v12, v13
	v_cvt_pk_f16_f32 v13, v14, v15
	v_or_b32_e32 v11, 0xa0, v2
	v_mfma_f32_16x16x32_f16 v[36:39], a[24:27], a[4:7], v[36:39]
	v_cvt_pk_f16_f32 v4, v4, v5
	v_cvt_pk_f16_f32 v5, v6, v7
	v_or_b32_e32 v6, 64, v2
	v_mfma_f32_16x16x32_f16 v[156:159], a[92:95], a[124:127], v[156:159]
	v_or_b32_e32 v7, 0x60, v2
	v_or_b32_e32 v14, 0xc0, v2
	v_or_b32_e32 v15, 0xe0, v2
	v_or_b32_e32 v19, 32, v18
	v_or_b32_e32 v22, 64, v18
	v_or_b32_e32 v23, 0x60, v18
	v_or_b32_e32 v26, 0x80, v18
	v_or_b32_e32 v27, 0xa0, v18
	v_or_b32_e32 v30, 0xc0, v18
	v_or_b32_e32 v31, 0xe0, v18
	s_add_u32 s36, s1, s2
	v_mul_lo_u32 v34, v2, s0
	v_cmp_gt_i32_e64 s[28:29], s14, v2
	v_cmp_gt_i32_e64 s[30:31], s15, v18
	v_cvt_pk_f16_f32 v40, v40, v41
	v_cvt_pk_f16_f32 v41, v42, v43
	v_cvt_pk_f16_f32 v36, v36, v37
	v_cvt_pk_f16_f32 v37, v38, v39
	s_addc_u32 s33, s4, s3
	v_mul_lo_u32 v35, v3, s0
	v_mul_lo_u32 v38, v6, s0
	v_mul_lo_u32 v39, v7, s0
	v_mul_lo_u32 v42, v10, s0
	v_mul_lo_u32 v43, v11, s0
	v_mul_lo_u32 v46, v14, s0
	v_mul_lo_u32 v47, v15, s0
	v_cmp_gt_i32_e64 s[26:27], s14, v3
	v_cmp_gt_i32_e64 s[24:25], s14, v6
	v_cmp_gt_i32_e64 s[22:23], s14, v7
	v_cmp_gt_i32_e64 s[20:21], s14, v10
	v_cmp_gt_i32_e64 s[18:19], s14, v11
	v_cmp_gt_i32_e64 s[16:17], s14, v14
	v_cmp_gt_i32_e32 vcc, s14, v15
	v_cmp_gt_i32_e64 s[12:13], s15, v19
	v_cmp_gt_i32_e64 s[10:11], s15, v22
	v_cmp_gt_i32_e64 s[8:9], s15, v23
	v_cmp_gt_i32_e64 s[6:7], s15, v26
	v_cmp_gt_i32_e64 s[4:5], s15, v27
	v_cmp_gt_i32_e64 s[2:3], s15, v30
	v_cmp_gt_i32_e64 s[0:1], s15, v31
	v_add_lshl_u32 v2, v18, v34, 1
	v_bfrev_b32_e32 v3, 1
	s_and_b64 s[14:15], s[28:29], s[30:31]
	s_and_b32 s37, s33, 0xffff
	v_cndmask_b32_e64 v2, v3, v2, s[14:15]
	buffer_store_dwordx2 v[194:195], v2, s[36:39], 0 offen
	v_add_lshl_u32 v2, v19, v34, 1
	s_and_b64 s[14:15], s[28:29], s[12:13]
	v_cvt_pk_f16_f32 v192, v156, v157
	v_cvt_pk_f16_f32 v193, v158, v159
	v_cndmask_b32_e64 v2, v3, v2, s[14:15]
	buffer_store_dwordx2 v[192:193], v2, s[36:39], 0 offen
	v_add_lshl_u32 v2, v22, v34, 1
	s_and_b64 s[14:15], s[28:29], s[10:11]
	v_cndmask_b32_e64 v2, v3, v2, s[14:15]
	buffer_store_dwordx2 v[248:249], v2, s[36:39], 0 offen
	v_add_lshl_u32 v2, v23, v34, 1
	s_and_b64 s[14:15], s[28:29], s[8:9]
	v_mfma_f32_16x16x32_f16 v[132:135], a[44:47], a[120:123], v[132:135]
	v_cndmask_b32_e64 v2, v3, v2, s[14:15]
	buffer_store_dwordx2 v[244:245], v2, s[36:39], 0 offen
	v_add_lshl_u32 v2, v26, v34, 1
	s_and_b64 s[14:15], s[28:29], s[6:7]
	v_mfma_f32_16x16x32_f16 v[124:127], a[24:27], a[124:127], v[124:127]
	v_cndmask_b32_e64 v2, v3, v2, s[14:15]
	buffer_store_dwordx2 v[242:243], v2, s[36:39], 0 offen
	v_add_lshl_u32 v2, v27, v34, 1
	s_and_b64 s[14:15], s[28:29], s[4:5]
	v_mfma_f32_16x16x32_f16 v[216:219], a[88:91], a[60:63], v[216:219]
	v_cndmask_b32_e64 v2, v3, v2, s[14:15]
	buffer_store_dwordx2 v[184:185], v2, s[36:39], 0 offen
	v_add_lshl_u32 v2, v30, v34, 1
	v_mfma_f32_16x16x32_f16 v[132:135], a[8:11], a[124:127], v[132:135]
	s_and_b64 s[14:15], s[28:29], s[2:3]
	v_cvt_pk_f16_f32 v240, v124, v125
	v_cvt_pk_f16_f32 v241, v126, v127
	v_cndmask_b32_e64 v2, v3, v2, s[14:15]
	buffer_store_dwordx2 v[240:241], v2, s[36:39], 0 offen
	v_add_lshl_u32 v2, v31, v34, 1
	s_and_b64 s[14:15], s[28:29], s[0:1]
	v_mfma_f32_16x16x32_f16 v[216:219], a[92:95], a[36:39], v[216:219]
	v_cvt_pk_f16_f32 v234, v132, v133
	v_cvt_pk_f16_f32 v235, v134, v135
	v_cndmask_b32_e64 v2, v3, v2, s[14:15]
	v_mfma_f32_16x16x32_f16 v[164:167], a[80:83], a[60:63], v[208:211]
	buffer_store_dwordx2 v[234:235], v2, s[36:39], 0 offen
	v_add_lshl_u32 v2, v35, v18, 1
	s_and_b64 s[14:15], s[26:27], s[30:31]
	v_cndmask_b32_e64 v2, v3, v2, s[14:15]
	buffer_store_dwordx2 v[232:233], v2, s[36:39], 0 offen
	v_add_lshl_u32 v2, v19, v35, 1
	s_and_b64 s[14:15], s[26:27], s[12:13]
	v_mfma_f32_16x16x32_f16 v[200:203], a[112:115], a[60:63], v[200:203]
	v_cvt_pk_f16_f32 v222, v216, v217
	v_cvt_pk_f16_f32 v223, v218, v219
	v_cndmask_b32_e64 v2, v3, v2, s[14:15]
	v_mfma_f32_16x16x32_f16 v[164:167], a[84:87], a[36:39], v[164:167]
	buffer_store_dwordx2 v[222:223], v2, s[36:39], 0 offen
	v_add_lshl_u32 v2, v22, v35, 1
	s_and_b64 s[14:15], s[26:27], s[10:11]
	v_mfma_f32_16x16x32_f16 v[112:115], a[16:19], a[36:39], v[112:115]
	v_cndmask_b32_e64 v2, v3, v2, s[14:15]
	buffer_store_dwordx2 v[220:221], v2, s[36:39], 0 offen
	v_add_lshl_u32 v2, v23, v35, 1
	v_mfma_f32_16x16x32_f16 v[188:191], a[28:31], a[60:63], v[188:191]
	s_and_b64 s[14:15], s[26:27], s[8:9]
	v_cvt_pk_f16_f32 v218, v164, v165
	v_cvt_pk_f16_f32 v219, v166, v167
	v_mfma_f32_16x16x32_f16 v[200:203], a[116:119], a[36:39], v[200:203]
	v_cndmask_b32_e64 v2, v3, v2, s[14:15]
	buffer_store_dwordx2 v[218:219], v2, s[36:39], 0 offen
	v_add_lshl_u32 v2, v26, v35, 1
	s_and_b64 s[14:15], s[26:27], s[6:7]
	v_mfma_f32_16x16x32_f16 v[100:103], a[28:31], a[64:67], v[100:103]
	v_cvt_pk_f16_f32 v166, v112, v113
	v_cvt_pk_f16_f32 v167, v114, v115
	v_cndmask_b32_e64 v2, v3, v2, s[14:15]
	v_mfma_f32_16x16x32_f16 v[144:147], a[80:83], a[72:75], v[144:147]
	buffer_store_dwordx2 v[166:167], v2, s[36:39], 0 offen
	v_add_lshl_u32 v2, v27, v35, 1
	s_and_b64 s[14:15], s[26:27], s[4:5]
	v_mfma_f32_16x16x32_f16 v[112:115], a[24:27], a[36:39], v[188:191]
	v_cvt_pk_f16_f32 v164, v200, v201
	v_cvt_pk_f16_f32 v165, v202, v203
	v_cndmask_b32_e64 v2, v3, v2, s[14:15]
	v_mfma_f32_16x16x32_f16 v[140:143], a[8:11], a[36:39], v[140:143]
	buffer_store_dwordx2 v[164:165], v2, s[36:39], 0 offen
	v_add_lshl_u32 v2, v30, v35, 1
	s_and_b64 s[14:15], s[26:27], s[2:3]
	v_mfma_f32_16x16x32_f16 v[236:239], a[24:27], a[68:71], v[100:103]
	v_cvt_pk_f16_f32 v216, v112, v113
	v_cvt_pk_f16_f32 v217, v114, v115
	v_cndmask_b32_e64 v2, v3, v2, s[14:15]
	v_mfma_f32_16x16x32_f16 v[100:103], a[56:59], a[104:107], v[180:183]
	buffer_store_dwordx2 v[216:217], v2, s[36:39], 0 offen
	v_add_lshl_u32 v2, v31, v35, 1
	s_and_b64 s[14:15], s[26:27], s[0:1]
	v_mfma_f32_16x16x32_f16 v[212:215], a[84:87], a[76:79], v[144:147]
	v_cvt_pk_f16_f32 v202, v140, v141
	v_cvt_pk_f16_f32 v203, v142, v143
	v_cndmask_b32_e64 v2, v3, v2, s[14:15]
	v_mfma_f32_16x16x32_f16 v[144:147], a[44:47], a[104:107], v[160:163]
	buffer_store_dwordx2 v[202:203], v2, s[36:39], 0 offen
	v_add_lshl_u32 v2, v38, v18, 1
	s_and_b64 s[14:15], s[24:25], s[30:31]
	v_mfma_f32_16x16x32_f16 v[160:163], a[44:47], a[64:67], v[148:151]
	v_cvt_pk_f16_f32 v158, v128, v129
	v_cvt_pk_f16_f32 v159, v130, v131
	v_cndmask_b32_e64 v2, v3, v2, s[14:15]
	v_mfma_f32_16x16x32_f16 v[176:179], a[80:83], a[104:107], v[176:179]
	buffer_store_dwordx2 v[158:159], v2, s[36:39], 0 offen
	v_add_lshl_u32 v2, v19, v38, 1
	s_and_b64 s[14:15], s[24:25], s[12:13]
	v_mfma_f32_16x16x32_f16 v[100:103], a[12:15], a[108:111], v[100:103]
	v_cvt_pk_f16_f32 v156, v196, v197
	v_cvt_pk_f16_f32 v157, v198, v199
	v_cndmask_b32_e64 v2, v3, v2, s[14:15]
	v_mfma_f32_16x16x32_f16 v[148:151], a[8:11], a[68:71], v[160:163]
	buffer_store_dwordx2 v[156:157], v2, s[36:39], 0 offen
	v_add_lshl_u32 v2, v22, v38, 1
	s_and_b64 s[14:15], s[24:25], s[10:11]
	v_mfma_f32_16x16x32_f16 v[160:163], a[52:55], a[104:107], v[172:175]
	v_cvt_pk_f16_f32 v200, v100, v101
	v_cvt_pk_f16_f32 v201, v102, v103
	v_cndmask_b32_e64 v2, v3, v2, s[14:15]
	v_mfma_f32_16x16x32_f16 v[168:171], a[112:115], a[104:107], v[168:171]
	buffer_store_dwordx2 v[200:201], v2, s[36:39], 0 offen
	v_add_lshl_u32 v2, v23, v38, 1
	s_and_b64 s[14:15], s[24:25], s[8:9]
	v_mfma_f32_16x16x32_f16 v[208:211], a[84:87], a[108:111], v[176:179]
	v_cndmask_b32_e64 v2, v3, v2, s[14:15]
	s_and_b64 s[14:15], s[24:25], s[6:7]
	v_cvt_pk_f16_f32 v142, v228, v229
	v_mfma_f32_16x16x32_f16 v[160:163], a[16:19], a[108:111], v[160:163]
	v_accvgpr_read_b32 v179, a163
	v_accvgpr_read_b32 v178, a162
	v_accvgpr_read_b32 v177, a161
	v_mfma_f32_16x16x32_f16 v[168:171], a[116:119], a[108:111], v[168:171]
	v_cvt_pk_f16_f32 v198, v208, v209
	v_cvt_pk_f16_f32 v199, v210, v211
	buffer_store_dwordx2 v[198:199], v2, s[36:39], 0 offen
	v_add_lshl_u32 v2, v26, v38, 1
	v_cvt_pk_f16_f32 v196, v160, v161
	v_cvt_pk_f16_f32 v197, v162, v163
	v_cndmask_b32_e64 v2, v3, v2, s[14:15]
	buffer_store_dwordx2 v[196:197], v2, s[36:39], 0 offen
	v_add_lshl_u32 v2, v27, v38, 1
	s_and_b64 s[14:15], s[24:25], s[4:5]
	v_mfma_f32_16x16x32_f16 v[152:155], a[88:91], a[72:75], v[152:155]
	v_cvt_pk_f16_f32 v168, v168, v169
	v_cvt_pk_f16_f32 v169, v170, v171
	v_cndmask_b32_e64 v2, v3, v2, s[14:15]
	v_mfma_f32_16x16x32_f16 v[144:147], a[8:11], a[108:111], v[144:147]
	v_accvgpr_read_b32 v176, a160
	buffer_store_dwordx2 v[168:169], v2, s[36:39], 0 offen
	v_add_lshl_u32 v2, v30, v38, 1
	s_and_b64 s[14:15], s[24:25], s[2:3]
	v_mfma_f32_16x16x32_f16 v[176:179], a[56:59], a[72:75], v[176:179]
	v_cvt_pk_f16_f32 v143, v230, v231
	v_cndmask_b32_e64 v2, v3, v2, s[14:15]
	buffer_store_dwordx2 v[142:143], v2, s[36:39], 0 offen
	v_add_lshl_u32 v2, v31, v38, 1
	s_and_b64 s[14:15], s[24:25], s[0:1]
	v_mfma_f32_16x16x32_f16 v[152:155], a[92:95], a[76:79], v[152:155]
	v_cvt_pk_f16_f32 v140, v144, v145
	v_cvt_pk_f16_f32 v141, v146, v147
	v_cndmask_b32_e64 v2, v3, v2, s[14:15]
	buffer_store_dwordx2 v[140:141], v2, s[36:39], 0 offen
	v_add_lshl_u32 v2, v39, v18, 1
	s_and_b64 s[14:15], s[22:23], s[30:31]
	v_mfma_f32_16x16x32_f16 v[176:179], a[12:15], a[76:79], v[176:179]
	v_cvt_pk_f16_f32 v134, v96, v97
	v_cvt_pk_f16_f32 v135, v98, v99
	v_cndmask_b32_e64 v2, v3, v2, s[14:15]
	buffer_store_dwordx2 v[134:135], v2, s[36:39], 0 offen
	v_add_lshl_u32 v2, v19, v39, 1
	s_and_b64 s[14:15], s[22:23], s[12:13]
	v_mfma_f32_16x16x32_f16 v[136:139], a[112:115], a[72:75], v[136:139]
	v_cvt_pk_f16_f32 v132, v152, v153
	v_cvt_pk_f16_f32 v133, v154, v155
	v_cndmask_b32_e64 v2, v3, v2, s[14:15]
	buffer_store_dwordx2 v[132:133], v2, s[36:39], 0 offen
	v_add_lshl_u32 v2, v22, v39, 1
	s_and_b64 s[14:15], s[22:23], s[10:11]
	v_accvgpr_read_b32 v183, a155
	v_cvt_pk_f16_f32 v130, v176, v177
	v_cvt_pk_f16_f32 v131, v178, v179
	v_cndmask_b32_e64 v2, v3, v2, s[14:15]
	v_accvgpr_read_b32 v182, a154
	v_accvgpr_read_b32 v181, a153
	v_accvgpr_read_b32 v180, a152
	buffer_store_dwordx2 v[130:131], v2, s[36:39], 0 offen
	v_add_lshl_u32 v2, v23, v39, 1
	s_and_b64 s[14:15], s[22:23], s[8:9]
	v_mfma_f32_16x16x32_f16 v[136:139], a[116:119], a[76:79], v[136:139]
	v_cvt_pk_f16_f32 v128, v212, v213
	v_cvt_pk_f16_f32 v129, v214, v215
	v_cndmask_b32_e64 v2, v3, v2, s[14:15]
	v_mfma_f32_16x16x32_f16 v[180:183], a[44:47], a[72:75], v[180:183]
	buffer_store_dwordx2 v[128:129], v2, s[36:39], 0 offen
	v_add_lshl_u32 v2, v26, v39, 1
	s_and_b64 s[14:15], s[22:23], s[6:7]
	v_cvt_pk_f16_f32 v126, v204, v205
	v_cvt_pk_f16_f32 v127, v206, v207
	v_cndmask_b32_e64 v2, v3, v2, s[14:15]
	buffer_store_dwordx2 v[126:127], v2, s[36:39], 0 offen
	v_add_lshl_u32 v2, v27, v39, 1
	s_and_b64 s[14:15], s[22:23], s[4:5]
	v_mfma_f32_16x16x32_f16 v[120:123], a[88:91], a[64:67], v[120:123]
	v_cvt_pk_f16_f32 v124, v136, v137
	v_cvt_pk_f16_f32 v125, v138, v139
	v_cndmask_b32_e64 v2, v3, v2, s[14:15]
	v_mfma_f32_16x16x32_f16 v[180:183], a[8:11], a[76:79], v[180:183]
	buffer_store_dwordx2 v[124:125], v2, s[36:39], 0 offen
	v_add_lshl_u32 v2, v30, v39, 1
	s_and_b64 s[14:15], s[22:23], s[2:3]
	v_cndmask_b32_e64 v2, v3, v2, s[14:15]
	buffer_store_dwordx2 v[118:119], v2, s[36:39], 0 offen
	v_add_lshl_u32 v2, v31, v39, 1
	s_and_b64 s[14:15], s[22:23], s[0:1]
	v_mfma_f32_16x16x32_f16 v[120:123], a[92:95], a[68:71], v[120:123]
	v_accvgpr_read_b32 v175, a159
	v_cvt_pk_f16_f32 v116, v180, v181
	v_cvt_pk_f16_f32 v117, v182, v183
	v_cndmask_b32_e64 v2, v3, v2, s[14:15]
	v_accvgpr_read_b32 v174, a158
	v_accvgpr_read_b32 v173, a157
	v_accvgpr_read_b32 v172, a156
	buffer_store_dwordx2 v[116:117], v2, s[36:39], 0 offen
	v_add_lshl_u32 v2, v42, v18, 1
	s_and_b64 s[14:15], s[20:21], s[30:31]
	v_mfma_f32_16x16x32_f16 v[172:175], a[52:55], a[64:67], v[172:175]
	v_cvt_pk_f16_f32 v114, v108, v109
	v_cvt_pk_f16_f32 v115, v110, v111
	v_cndmask_b32_e64 v2, v3, v2, s[14:15]
	buffer_store_dwordx2 v[114:115], v2, s[36:39], 0 offen
	v_add_lshl_u32 v2, v19, v42, 1
	s_and_b64 s[14:15], s[20:21], s[12:13]
	v_mfma_f32_16x16x32_f16 v[104:107], a[112:115], a[64:67], v[104:107]
	v_cvt_pk_f16_f32 v112, v120, v121
	v_cvt_pk_f16_f32 v113, v122, v123
	v_cndmask_b32_e64 v2, v3, v2, s[14:15]
	buffer_store_dwordx2 v[112:113], v2, s[36:39], 0 offen
	v_add_lshl_u32 v2, v22, v42, 1
	s_and_b64 s[14:15], s[20:21], s[10:11]
	v_mfma_f32_16x16x32_f16 v[172:175], a[16:19], a[68:71], v[172:175]
	v_cvt_pk_f16_f32 v110, v224, v225
	v_cvt_pk_f16_f32 v111, v226, v227
	v_cndmask_b32_e64 v2, v3, v2, s[14:15]
	buffer_store_dwordx2 v[110:111], v2, s[36:39], 0 offen
	v_add_lshl_u32 v2, v23, v42, 1
	s_and_b64 s[14:15], s[20:21], s[8:9]
	v_mfma_f32_16x16x32_f16 v[104:107], a[116:119], a[68:71], v[104:107]
	v_cvt_pk_f16_f32 v108, v252, v253
	v_cvt_pk_f16_f32 v109, v254, v255
	v_cndmask_b32_e64 v2, v3, v2, s[14:15]
	buffer_store_dwordx2 v[108:109], v2, s[36:39], 0 offen
	v_add_lshl_u32 v2, v26, v42, 1
	s_and_b64 s[14:15], s[20:21], s[6:7]
	v_mfma_f32_16x16x32_f16 v[92:95], a[96:99], a[32:35], v[92:95]
	v_cvt_pk_f16_f32 v102, v172, v173
	v_cvt_pk_f16_f32 v103, v174, v175
	v_cndmask_b32_e64 v2, v3, v2, s[14:15]
	buffer_store_dwordx2 v[102:103], v2, s[36:39], 0 offen
	v_add_lshl_u32 v2, v27, v42, 1
	s_and_b64 s[14:15], s[20:21], s[4:5]
	v_mfma_f32_16x16x32_f16 v[88:91], a[88:91], a[32:35], v[88:91]
	v_cvt_pk_f16_f32 v100, v104, v105
	v_cvt_pk_f16_f32 v101, v106, v107
	v_cndmask_b32_e64 v2, v3, v2, s[14:15]
	buffer_store_dwordx2 v[100:101], v2, s[36:39], 0 offen
	v_add_lshl_u32 v2, v30, v42, 1
	s_and_b64 s[14:15], s[20:21], s[2:3]
	v_mfma_f32_16x16x32_f16 v[92:95], a[100:103], a[20:23], v[92:95]
	v_cvt_pk_f16_f32 v98, v236, v237
	v_cvt_pk_f16_f32 v99, v238, v239
	v_cndmask_b32_e64 v2, v3, v2, s[14:15]
	buffer_store_dwordx2 v[98:99], v2, s[36:39], 0 offen
	v_add_lshl_u32 v2, v31, v42, 1
	s_and_b64 s[14:15], s[20:21], s[0:1]
	v_mfma_f32_16x16x32_f16 v[88:91], a[92:95], a[20:23], v[88:91]
	v_cvt_pk_f16_f32 v96, v148, v149
	v_cvt_pk_f16_f32 v97, v150, v151
	v_cndmask_b32_e64 v2, v3, v2, s[14:15]
	v_mfma_f32_16x16x32_f16 v[80:83], a[80:83], a[32:35], v[80:83]
	buffer_store_dwordx2 v[96:97], v2, s[36:39], 0 offen
	v_add_lshl_u32 v2, v43, v18, 1
	s_and_b64 s[14:15], s[18:19], s[30:31]
	v_mfma_f32_16x16x32_f16 v[76:79], a[52:55], a[32:35], v[76:79]
	v_cvt_pk_f16_f32 v92, v92, v93
	v_cvt_pk_f16_f32 v93, v94, v95
	v_cndmask_b32_e64 v2, v3, v2, s[14:15]
	buffer_store_dwordx2 v[92:93], v2, s[36:39], 0 offen
	v_add_lshl_u32 v2, v19, v43, 1
	s_and_b64 s[14:15], s[18:19], s[12:13]
	v_mfma_f32_16x16x32_f16 v[72:75], a[112:115], a[32:35], v[72:75]
	v_cvt_pk_f16_f32 v88, v88, v89
	v_cvt_pk_f16_f32 v89, v90, v91
	v_cndmask_b32_e64 v2, v3, v2, s[14:15]
	v_mfma_f32_16x16x32_f16 v[80:83], a[84:87], a[20:23], v[80:83]
	buffer_store_dwordx2 v[88:89], v2, s[36:39], 0 offen
	v_add_lshl_u32 v2, v22, v43, 1
	s_and_b64 s[14:15], s[18:19], s[10:11]
	v_mfma_f32_16x16x32_f16 v[76:79], a[16:19], a[20:23], v[76:79]
	v_cndmask_b32_e64 v2, v3, v2, s[14:15]
	buffer_store_dwordx2 v[84:85], v2, s[36:39], 0 offen
	v_add_lshl_u32 v2, v23, v43, 1
	v_mfma_f32_16x16x32_f16 v[68:71], a[28:31], a[32:35], v[68:71]
	s_and_b64 s[14:15], s[18:19], s[8:9]
	v_cvt_pk_f16_f32 v80, v80, v81
	v_cvt_pk_f16_f32 v81, v82, v83
	v_mfma_f32_16x16x32_f16 v[72:75], a[116:119], a[20:23], v[72:75]
	v_cndmask_b32_e64 v2, v3, v2, s[14:15]
	buffer_store_dwordx2 v[80:81], v2, s[36:39], 0 offen
	v_add_lshl_u32 v2, v26, v43, 1
	s_and_b64 s[14:15], s[18:19], s[6:7]
	v_mfma_f32_16x16x32_f16 v[60:63], a[96:99], a[40:43], v[60:63]
	v_cvt_pk_f16_f32 v76, v76, v77
	v_cvt_pk_f16_f32 v77, v78, v79
	v_cndmask_b32_e64 v2, v3, v2, s[14:15]
	v_mfma_f32_16x16x32_f16 v[68:71], a[24:27], a[20:23], v[68:71]
	buffer_store_dwordx2 v[76:77], v2, s[36:39], 0 offen
	v_add_lshl_u32 v2, v27, v43, 1
	s_and_b64 s[14:15], s[18:19], s[4:5]
	v_mfma_f32_16x16x32_f16 v[56:59], a[88:91], a[40:43], v[56:59]
	v_cvt_pk_f16_f32 v72, v72, v73
	v_cvt_pk_f16_f32 v73, v74, v75
	v_cndmask_b32_e64 v2, v3, v2, s[14:15]
	buffer_store_dwordx2 v[72:73], v2, s[36:39], 0 offen
	v_add_lshl_u32 v2, v30, v43, 1
	s_and_b64 s[14:15], s[18:19], s[2:3]
	v_mfma_f32_16x16x32_f16 v[60:63], a[100:103], a[4:7], v[60:63]
	v_cvt_pk_f16_f32 v68, v68, v69
	v_cvt_pk_f16_f32 v69, v70, v71
	v_cndmask_b32_e64 v2, v3, v2, s[14:15]
	v_mfma_f32_16x16x32_f16 v[52:55], a[56:59], a[40:43], v[52:55]
	buffer_store_dwordx2 v[68:69], v2, s[36:39], 0 offen
	v_add_lshl_u32 v2, v31, v43, 1
	s_and_b64 s[14:15], s[18:19], s[0:1]
	v_mfma_f32_16x16x32_f16 v[56:59], a[92:95], a[4:7], v[56:59]
	v_cndmask_b32_e64 v2, v3, v2, s[14:15]
	buffer_store_dwordx2 v[64:65], v2, s[36:39], 0 offen
	v_add_lshl_u32 v2, v46, v18, 1
	v_mfma_f32_16x16x32_f16 v[48:51], a[80:83], a[40:43], v[48:51]
	s_and_b64 s[14:15], s[16:17], s[30:31]
	v_cvt_pk_f16_f32 v60, v60, v61
	v_cvt_pk_f16_f32 v61, v62, v63
	v_mfma_f32_16x16x32_f16 v[52:55], a[12:15], a[4:7], v[52:55]
	v_cndmask_b32_e64 v2, v3, v2, s[14:15]
	buffer_store_dwordx2 v[60:61], v2, s[36:39], 0 offen
	v_add_lshl_u32 v2, v19, v46, 1
	s_and_b64 s[14:15], s[16:17], s[12:13]
	v_mfma_f32_16x16x32_f16 v[48:51], a[84:87], a[4:7], v[48:51]
	v_cvt_pk_f16_f32 v56, v56, v57
	v_cvt_pk_f16_f32 v57, v58, v59
	v_cndmask_b32_e64 v2, v3, v2, s[14:15]
	buffer_store_dwordx2 v[56:57], v2, s[36:39], 0 offen
	v_add_lshl_u32 v2, v22, v46, 1
	s_and_b64 s[14:15], s[16:17], s[10:11]
	v_cvt_pk_f16_f32 v52, v52, v53
	v_cvt_pk_f16_f32 v53, v54, v55
	v_cndmask_b32_e64 v2, v3, v2, s[14:15]
	buffer_store_dwordx2 v[52:53], v2, s[36:39], 0 offen
	v_add_lshl_u32 v2, v23, v46, 1
	s_and_b64 s[14:15], s[16:17], s[8:9]
	v_cvt_pk_f16_f32 v48, v48, v49
	v_cvt_pk_f16_f32 v49, v50, v51
	v_cndmask_b32_e64 v2, v3, v2, s[14:15]
	buffer_store_dwordx2 v[48:49], v2, s[36:39], 0 offen
	v_add_lshl_u32 v2, v26, v46, 1
	s_and_b64 s[14:15], s[16:17], s[6:7]
	v_cndmask_b32_e64 v2, v3, v2, s[14:15]
	buffer_store_dwordx2 v[44:45], v2, s[36:39], 0 offen
	v_add_lshl_u32 v2, v27, v46, 1
	s_and_b64 s[14:15], s[16:17], s[4:5]
	v_cndmask_b32_e64 v2, v3, v2, s[14:15]
	buffer_store_dwordx2 v[40:41], v2, s[36:39], 0 offen
	v_add_lshl_u32 v2, v30, v46, 1
	s_and_b64 s[14:15], s[16:17], s[2:3]
	v_cndmask_b32_e64 v2, v3, v2, s[14:15]
	buffer_store_dwordx2 v[36:37], v2, s[36:39], 0 offen
	v_add_lshl_u32 v2, v31, v46, 1
	s_and_b64 s[14:15], s[16:17], s[0:1]
	v_cndmask_b32_e64 v2, v3, v2, s[14:15]
	buffer_store_dwordx2 v[32:33], v2, s[36:39], 0 offen
	v_add_lshl_u32 v2, v47, v18, 1
	s_and_b64 s[14:15], vcc, s[30:31]
	v_cndmask_b32_e64 v2, v3, v2, s[14:15]
	buffer_store_dwordx2 v[28:29], v2, s[36:39], 0 offen
	v_add_lshl_u32 v2, v19, v47, 1
	s_and_b64 s[12:13], vcc, s[12:13]
	v_cndmask_b32_e64 v2, v3, v2, s[12:13]
	buffer_store_dwordx2 v[24:25], v2, s[36:39], 0 offen
	v_add_lshl_u32 v2, v22, v47, 1
	s_and_b64 s[10:11], vcc, s[10:11]
	v_cndmask_b32_e64 v2, v3, v2, s[10:11]
	buffer_store_dwordx2 v[20:21], v2, s[36:39], 0 offen
	v_add_lshl_u32 v2, v23, v47, 1
	s_and_b64 s[8:9], vcc, s[8:9]
	v_cndmask_b32_e64 v2, v3, v2, s[8:9]
	buffer_store_dwordx2 v[16:17], v2, s[36:39], 0 offen
	v_add_lshl_u32 v2, v26, v47, 1
	s_and_b64 s[6:7], vcc, s[6:7]
	v_cndmask_b32_e64 v2, v3, v2, s[6:7]
	buffer_store_dwordx2 v[12:13], v2, s[36:39], 0 offen
	v_add_lshl_u32 v2, v27, v47, 1
	s_and_b64 s[4:5], vcc, s[4:5]
	v_cndmask_b32_e64 v2, v3, v2, s[4:5]
	buffer_store_dwordx2 v[8:9], v2, s[36:39], 0 offen
	v_add_lshl_u32 v2, v30, v47, 1
	s_and_b64 s[2:3], vcc, s[2:3]
	v_cndmask_b32_e64 v2, v3, v2, s[2:3]
	buffer_store_dwordx2 v[4:5], v2, s[36:39], 0 offen
	v_add_lshl_u32 v2, v31, v47, 1
	s_and_b64 vcc, vcc, s[0:1]
	v_cndmask_b32_e32 v2, v3, v2, vcc
	buffer_store_dwordx2 v[0:1], v2, s[36:39], 0 offen
	s_endpgm
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel v5_local_prefetch
		.amdhsa_group_segment_fixed_size 0
		.amdhsa_private_segment_fixed_size 0
		.amdhsa_kernarg_size 320
		.amdhsa_user_sgpr_count 16
		.amdhsa_user_sgpr_dispatch_ptr 1
		.amdhsa_user_sgpr_queue_ptr 1
		.amdhsa_user_sgpr_kernarg_segment_ptr 1
		.amdhsa_user_sgpr_dispatch_id 1
		.amdhsa_user_sgpr_kernarg_preload_length 8
		.amdhsa_user_sgpr_kernarg_preload_offset 0
		.amdhsa_user_sgpr_private_segment_size 0
		.amdhsa_uses_dynamic_stack 0
		.amdhsa_enable_private_segment 0
		.amdhsa_system_sgpr_workgroup_id_x 1
		.amdhsa_system_sgpr_workgroup_id_y 1
		.amdhsa_system_sgpr_workgroup_id_z 1
		.amdhsa_system_sgpr_workgroup_info 0
		.amdhsa_system_vgpr_workitem_id 2
		.amdhsa_next_free_vgpr 452
		.amdhsa_next_free_sgpr 40
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
	.size	v5_local_prefetch, .Lfunc_end0-v5_local_prefetch
	.cfi_endproc
                                        ; -- End function
	.set v5_local_prefetch.num_vgpr, 256
	.set v5_local_prefetch.num_agpr, 196
	.set v5_local_prefetch.numbered_sgpr, 40
	.set v5_local_prefetch.num_named_barrier, 0
	.set v5_local_prefetch.private_seg_size, 0
	.set v5_local_prefetch.uses_vcc, 1
	.set v5_local_prefetch.uses_flat_scratch, 0
	.set v5_local_prefetch.has_dyn_sized_stack, 0
	.set v5_local_prefetch.has_recursion, 0
	.set v5_local_prefetch.has_indirect_call, 0
	.section	.AMDGPU.csdata,"",@progbits
; Kernel info:
; codeLenInByte = 14556
; TotalNumSgprs: 46
; NumVgprs: 256
; NumAgprs: 196
; TotalNumVgprs: 452
; ScratchSize: 0
; MemoryBound: 0
; FloatMode: 240
; IeeeMode: 1
; LDSByteSize: 0 bytes/workgroup (compile time only)
; SGPRBlocks: 5
; VGPRBlocks: 56
; NumSGPRsForWavesPerEU: 46
; NumVGPRsForWavesPerEU: 452
; AccumOffset: 256
; Occupancy: 1
; WaveLimiterHint : 0
; COMPUTE_PGM_RSRC2:SCRATCH_EN: 0
; COMPUTE_PGM_RSRC2:USER_SGPR: 16
; COMPUTE_PGM_RSRC2:TRAP_HANDLER: 0
; COMPUTE_PGM_RSRC2:TGID_X_EN: 1
; COMPUTE_PGM_RSRC2:TGID_Y_EN: 1
; COMPUTE_PGM_RSRC2:TGID_Z_EN: 1
; COMPUTE_PGM_RSRC2:TIDIG_COMP_CNT: 2
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
	.section	.debug_info,"",@progbits
.Lcu_begin0:
	.long	.Ldebug_info_end0-.Ldebug_info_start0 ; Length of Unit
.Ldebug_info_start0:
	.short	4                               ; DWARF version number
	.long	.debug_abbrev                   ; Offset Into Abbrev. Section
	.byte	8                               ; Address Size (in bytes)
	.byte	1                               ; Abbrev [1] 0xb:0x58 DW_TAG_compile_unit
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
	.byte	3                               ; Abbrev [3] 0x30:0x32 DW_TAG_subprogram
	.quad	.Lfunc_begin0                   ; DW_AT_low_pc
	.long	.Lfunc_end0-.Lfunc_begin0       ; DW_AT_high_pc
	.long	42                              ; DW_AT_abstract_origin
	.byte	4                               ; Abbrev [4] 0x41:0x14 DW_TAG_inlined_subroutine
	.long	42                              ; DW_AT_abstract_origin
	.quad	.Ltmp1                          ; DW_AT_low_pc
	.long	.Ltmp2-.Ltmp1                   ; DW_AT_high_pc
	.byte	1                               ; DW_AT_call_file
	.byte	77                              ; DW_AT_call_line
	.byte	17                              ; DW_AT_call_column
	.byte	5                               ; Abbrev [5] 0x55:0xc DW_TAG_inlined_subroutine
	.long	42                              ; DW_AT_abstract_origin
	.long	.Ldebug_ranges0                 ; DW_AT_ranges
	.byte	1                               ; DW_AT_call_file
	.byte	169                             ; DW_AT_call_line
	.byte	15                              ; DW_AT_call_column
	.byte	0                               ; End Of Children Mark
	.byte	0                               ; End Of Children Mark
.Ldebug_info_end0:
	.section	.debug_ranges,"",@progbits
.Ldebug_ranges0:
	.quad	.Ltmp3-.Lfunc_begin0
	.quad	.Ltmp4-.Lfunc_begin0
	.quad	.Ltmp5-.Lfunc_begin0
	.quad	.Ltmp6-.Lfunc_begin0
	.quad	0
	.quad	0
	.section	.debug_str,"MS",@progbits,1
.Linfo_string0:
	.asciz	"triton"                        ; string offset=0 ; triton
.Linfo_string1:
	.asciz	"matmul_kernel.py"              ; string offset=7 ; matmul_kernel.py
.Linfo_string2:
	.asciz	"kernels/gemm/a16w16/v5_local_prefetch" ; string offset=24 ; kernels/gemm/a16w16/v5_local_prefetch
.Linfo_string3:
	.asciz	"v5_local_prefetch"             ; string offset=96 ; v5_local_prefetch
	.section	".note.GNU-stack","",@progbits
	.amdgpu_metadata
---
amdhsa.kernels:
  - .agpr_count:     196
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
      - .offset:         64
        .size:           4
        .value_kind:     hidden_block_count_x
      - .offset:         68
        .size:           4
        .value_kind:     hidden_block_count_y
      - .offset:         72
        .size:           4
        .value_kind:     hidden_block_count_z
      - .offset:         76
        .size:           2
        .value_kind:     hidden_group_size_x
      - .offset:         78
        .size:           2
        .value_kind:     hidden_group_size_y
      - .offset:         80
        .size:           2
        .value_kind:     hidden_group_size_z
      - .offset:         82
        .size:           2
        .value_kind:     hidden_remainder_x
      - .offset:         84
        .size:           2
        .value_kind:     hidden_remainder_y
      - .offset:         86
        .size:           2
        .value_kind:     hidden_remainder_z
      - .offset:         104
        .size:           8
        .value_kind:     hidden_global_offset_x
      - .offset:         112
        .size:           8
        .value_kind:     hidden_global_offset_y
      - .offset:         120
        .size:           8
        .value_kind:     hidden_global_offset_z
      - .offset:         128
        .size:           2
        .value_kind:     hidden_grid_dims
      - .offset:         144
        .size:           8
        .value_kind:     hidden_hostcall_buffer
      - .offset:         152
        .size:           8
        .value_kind:     hidden_multigrid_sync_arg
      - .offset:         160
        .size:           8
        .value_kind:     hidden_heap_v1
      - .offset:         168
        .size:           8
        .value_kind:     hidden_default_queue
      - .offset:         176
        .size:           8
        .value_kind:     hidden_completion_action
      - .offset:         184
        .size:           4
        .value_kind:     hidden_dynamic_lds_size
      - .offset:         264
        .size:           8
        .value_kind:     hidden_queue_ptr
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 320
    .max_flat_workgroup_size: 256
    .name:           v5_local_prefetch
    .private_segment_fixed_size: 0
    .sgpr_count:     46
    .sgpr_spill_count: 0
    .symbol:         v5_local_prefetch.kd
    .uniform_work_group_size: 1
    .uses_dynamic_stack: false
    .vgpr_count:     452
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
