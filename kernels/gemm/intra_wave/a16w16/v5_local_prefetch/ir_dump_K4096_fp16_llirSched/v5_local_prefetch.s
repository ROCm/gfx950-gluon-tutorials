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
	s_mov_b32 s1, s15
	s_load_dwordx2 s[18:19], s[4:5], 0x20
	s_load_dword s17, s[4:5], 0x28
	v_and_b32_e32 v1, 0x3ff, v0
	s_nop 0
	v_readfirstlane_b32 s3, v1
	s_bfe_u32 s20, s3, 0x20006
	.file	2 "python/triton/language" "standard.py"
	s_add_i32 s0, s15, 0xff
	s_ashr_i32 s2, s0, 31
	s_lshr_b32 s2, s2, 24
	s_add_i32 s0, s0, s2
	s_ashr_i32 s0, s0, 8
	s_xor_b32 s2, s16, s0
	s_ashr_i32 s2, s2, 31
	s_abs_i32 s6, s16
	s_abs_i32 s7, s0
	v_cvt_f32_u32_e32 v2, s7
	v_rcp_iflag_f32_e32 v2, v2
	s_nop 0
	v_mul_f32_e32 v2, 0x4f7ffffe, v2
	v_cvt_u32_f32_e32 v2, v2
	s_mov_b32 s15, 0
	s_sub_i32 s21, 0, s7
	v_readfirstlane_b32 s22, v2
	s_mul_i32 s21, s21, s22
	s_mul_hi_u32 s21, s22, s21
	s_add_i32 s22, s22, s21
	s_mul_hi_u32 s21, s6, s22
	s_mul_i32 s22, s21, s7
	s_sub_i32 s6, s6, s22
	s_add_i32 s22, s21, 1
	s_sub_i32 s23, s6, s7
	s_cmp_ge_u32 s6, s7
	s_cselect_b32 s21, s22, s21
	s_cselect_b32 s6, s23, s6
	s_add_i32 s22, s21, 1
	s_cmp_ge_u32 s6, s7
	s_cselect_b32 s6, s22, s21
	s_xor_b32 s6, s6, s2
	s_sub_i32 s2, s6, s2
	s_mul_i32 s0, s2, s0
	s_sub_i32 s21, s16, s0
	v_and_b32_e32 v2, 63, v0
	v_lshl_or_b32 v18, s20, 6, v2
	v_lshlrev_b32_e32 v2, 1, v1
	v_and_b32_e32 v2, 0x70, v2
	v_or_b32_e32 v2, s20, v2
	v_or_b32_e32 v3, 4, v2
	v_or_b32_e32 v4, 8, v2
	v_or_b32_e32 v5, 12, v2
	v_or_b32_e32 v6, 0x80, v2
	v_or_b32_e32 v7, 0x84, v2
	v_or_b32_e32 v8, 0x88, v2
	v_or_b32_e32 v9, 0x8c, v2
	v_lshlrev_b32_e32 v1, 3, v1
	v_and_b32_e32 v1, 56, v1
	s_lshl_b32 s0, s2, 8
	s_waitcnt lgkmcnt(0)
	s_mul_i32 s6, s0, s19
	s_ashr_i32 s7, s6, 31
	s_lshl_b64 s[6:7], s[6:7], 1
	s_add_u32 s16, s8, s6
	s_addc_u32 s7, s9, s7
	s_lshl_b32 s2, s21, 8
	s_mul_i32 s8, s2, s17
	s_ashr_i32 s9, s8, 31
	s_lshl_b64 s[8:9], s[8:9], 1
	s_add_u32 s8, s10, s8
	s_addc_u32 s24, s11, s9
	v_mul_lo_u32 v10, v2, s19
	v_mul_lo_u32 v11, v3, s19
	v_mul_lo_u32 v12, v4, s19
	v_mul_lo_u32 v13, v5, s19
	v_mul_lo_u32 v14, v6, s19
	v_mul_lo_u32 v15, v7, s19
	v_mul_lo_u32 v16, v8, s19
	v_mul_lo_u32 v17, v9, s19
	v_mul_lo_u32 v2, v2, s17
	v_mul_lo_u32 v3, v3, s17
	v_mul_lo_u32 v4, v4, s17
	v_mul_lo_u32 v5, v5, s17
	v_mul_lo_u32 v6, v6, s17
	v_mul_lo_u32 v7, v7, s17
	v_mul_lo_u32 v8, v8, s17
	v_mul_lo_u32 v9, v9, s17
	s_add_i32 s25, s18, 63
	s_and_b32 s17, s7, 0xffff
	s_mov_b32 s19, 0x27000
	s_mov_b32 s18, 0x7ffffffe
	s_mul_i32 s6, s20, 0x420
	s_add_i32 s26, s6, 0
	v_add_lshl_u32 v252, v10, v1, 1
	s_mov_b32 m0, s26
	s_nop 0
	buffer_load_dwordx4 v252, s[16:19], 0 offen lds
	s_add_i32 m0, s26, 0x1080
	v_add_lshl_u32 v10, v11, v1, 1
	buffer_load_dwordx4 v10, s[16:19], 0 offen lds
	s_add_i32 m0, s26, 0x2100
	v_add_lshl_u32 v11, v12, v1, 1
	buffer_load_dwordx4 v11, s[16:19], 0 offen lds
	s_add_i32 m0, s26, 0x3180
	v_add_lshl_u32 v12, v13, v1, 1
	buffer_load_dwordx4 v12, s[16:19], 0 offen lds
	s_add_i32 m0, s26, 0x4200
	v_add_lshl_u32 v253, v14, v1, 1
	buffer_load_dwordx4 v253, s[16:19], 0 offen lds
	s_add_i32 m0, s26, 0x5280
	v_add_lshl_u32 v254, v15, v1, 1
	buffer_load_dwordx4 v254, s[16:19], 0 offen lds
	s_add_i32 m0, s26, 0x6300
	v_add_lshl_u32 v13, v16, v1, 1
	buffer_load_dwordx4 v13, s[16:19], 0 offen lds
	s_add_i32 m0, s26, 0x7380
	v_add_lshl_u32 v14, v17, v1, 1
	buffer_load_dwordx4 v14, s[16:19], 0 offen lds
	s_and_b32 s9, s24, 0xffff
	s_mov_b32 s10, s18
	s_mov_b32 s11, s19
	s_add_i32 s17, 0, 0x107e0
	s_add_i32 s21, s17, s6
	v_add_lshl_u32 v15, v2, v1, 1
	s_mov_b32 m0, s21
	s_nop 0
	buffer_load_dwordx4 v15, s[8:11], 0 offen lds
	s_add_i32 m0, s21, 0x1080
	v_add_lshl_u32 v16, v3, v1, 1
	buffer_load_dwordx4 v16, s[8:11], 0 offen lds
	s_add_i32 m0, s21, 0x2100
	v_add_lshl_u32 v4, v4, v1, 1
	buffer_load_dwordx4 v4, s[8:11], 0 offen lds
	s_add_i32 m0, s21, 0x3180
	v_add_lshl_u32 v5, v5, v1, 1
	buffer_load_dwordx4 v5, s[8:11], 0 offen lds
	s_add_i32 m0, s21, 0x4200
	v_add_lshl_u32 v6, v6, v1, 1
	buffer_load_dwordx4 v6, s[8:11], 0 offen lds
	s_add_i32 m0, s21, 0x5280
	v_add_lshl_u32 v7, v7, v1, 1
	buffer_load_dwordx4 v7, s[8:11], 0 offen lds
	s_add_i32 m0, s21, 0x6300
	v_add_lshl_u32 v8, v8, v1, 1
	buffer_load_dwordx4 v8, s[8:11], 0 offen lds
	s_add_i32 m0, s21, 0x7380
	v_add_lshl_u32 v9, v9, v1, 1
	buffer_load_dwordx4 v9, s[8:11], 0 offen lds
	; asyncmark
	s_add_u32 s28, s16, 0x80
	s_addc_u32 s9, s7, 0
	s_add_u32 s20, s8, 0x80
	s_addc_u32 s10, s24, 0
	s_barrier
	s_and_b32 s29, s9, 0xffff
	s_mov_b32 s30, s18
	s_mov_b32 s31, s19
	s_add_i32 m0, s21, 0xffff7c20
	s_nop 0
	buffer_load_dwordx4 v252, s[28:31], 0 offen lds
	s_add_i32 m0, s21, 0xffff8ca0
	s_nop 0
	buffer_load_dwordx4 v10, s[28:31], 0 offen lds
	s_add_i32 m0, s21, 0xffff9d20
	s_nop 0
	buffer_load_dwordx4 v11, s[28:31], 0 offen lds
	s_add_i32 m0, s21, 0xffffada0
	s_nop 0
	buffer_load_dwordx4 v12, s[28:31], 0 offen lds
	s_add_i32 m0, s21, 0xffffbe20
	s_nop 0
	buffer_load_dwordx4 v253, s[28:31], 0 offen lds
	s_add_i32 m0, s21, 0xffffcea0
	s_nop 0
	buffer_load_dwordx4 v254, s[28:31], 0 offen lds
	s_add_i32 m0, s21, 0xffffdf20
	s_nop 0
	buffer_load_dwordx4 v13, s[28:31], 0 offen lds
	s_add_i32 m0, s21, 0xffffefa0
	s_nop 0
	buffer_load_dwordx4 v14, s[28:31], 0 offen lds
	s_and_b32 s21, s10, 0xffff
	s_mov_b32 s22, s18
	s_mov_b32 s23, s19
	s_add_i32 m0, s26, 0x18be0
	s_nop 0
	buffer_load_dwordx4 v15, s[20:23], 0 offen lds
	s_add_i32 m0, s26, 0x19c60
	s_nop 0
	buffer_load_dwordx4 v16, s[20:23], 0 offen lds
	s_add_i32 m0, s26, 0x1ace0
	s_nop 0
	buffer_load_dwordx4 v4, s[20:23], 0 offen lds
	s_add_i32 m0, s26, 0x1bd60
	s_nop 0
	buffer_load_dwordx4 v5, s[20:23], 0 offen lds
	s_add_i32 m0, s26, 0x1cde0
	s_nop 0
	buffer_load_dwordx4 v6, s[20:23], 0 offen lds
	s_add_i32 m0, s26, 0x1de60
	s_nop 0
	buffer_load_dwordx4 v7, s[20:23], 0 offen lds
	s_add_i32 m0, s26, 0x1eee0
	s_nop 0
	buffer_load_dwordx4 v8, s[20:23], 0 offen lds
	s_add_i32 m0, s26, 0x1ff60
	s_nop 0
	buffer_load_dwordx4 v9, s[20:23], 0 offen lds
	; asyncmark
	; wait_asyncmark(1)
	s_waitcnt vmcnt(16)
	s_barrier
	v_and_b32_e32 v3, 15, v0
	v_lshlrev_b32_e32 v1, 10, v3
	s_movk_i32 s9, 0xb0
	v_accvgpr_write_b32 a208, v18
	v_and_or_b32 v2, v18, s9, v1
	v_accvgpr_write_b32 a209, v3
	v_lshlrev_b32_e32 v3, 5, v3
	v_add_u32_e32 v17, v2, v3
	v_add_u32_e32 v2, 0, v17
	ds_read_b128 a[120:123], v2
	ds_read_b128 a[124:127], v2 offset:64
	ds_read_b128 a[112:115], v2 offset:256
	ds_read_b128 a[116:119], v2 offset:320
	ds_read_b128 a[104:107], v2 offset:512
	ds_read_b128 a[108:111], v2 offset:576
	ds_read_b128 a[96:99], v2 offset:768
	ds_read_b128 a[100:103], v2 offset:832
	ds_read_b128 a[16:19], v2 offset:16896
	ds_read_b128 a[20:23], v2 offset:16960
	ds_read_b128 a[0:3], v2 offset:17152
	ds_read_b128 a[4:7], v2 offset:17216
	ds_read_b128 a[72:75], v2 offset:17408
	ds_read_b128 a[76:79], v2 offset:17472
	ds_read_b128 a[8:11], v2 offset:17664
	ds_read_b128 a[12:15], v2 offset:17728
	v_and_b32_e32 v0, 48, v0
	s_lshl_b32 s9, s3, 1
	s_and_b32 s9, s9, 0x80
	v_or3_b32 v0, v0, s9, v1
	v_add_u32_e32 v1, v0, v3
	v_add_u32_e32 v0, s17, v1
	ds_read_b128 a[88:91], v0
	ds_read_b128 a[92:95], v0 offset:64
	ds_read_b128 a[80:83], v0 offset:256
	ds_read_b128 a[84:87], v0 offset:320
	ds_read_b128 a[64:67], v0 offset:512
	ds_read_b128 a[68:71], v0 offset:576
	ds_read_b128 a[56:59], v0 offset:768
	ds_read_b128 a[60:63], v0 offset:832
	ds_read_b128 a[48:51], v0 offset:16896
	ds_read_b128 a[52:55], v0 offset:16960
	ds_read_b128 a[40:43], v0 offset:17152
	ds_read_b128 a[44:47], v0 offset:17216
	ds_read_b128 a[32:35], v0 offset:17408
	ds_read_b128 a[36:39], v0 offset:17472
	ds_read_b128 a[24:27], v0 offset:17664
	ds_read_b128 a[28:31], v0 offset:17728
	s_cmpk_lt_i32 s25, 0x80
	s_cbranch_scc1 .LBB0_5
; %bb.1:                                ; %.lr.ph
	v_accvgpr_write_b32 a225, v1
	v_accvgpr_write_b32 a224, v17
	v_accvgpr_write_b32 a221, v7
	v_accvgpr_write_b32 a220, v6
	v_accvgpr_write_b32 a219, v5
	v_accvgpr_write_b32 a218, v4
	v_accvgpr_write_b32 a217, v16
	v_accvgpr_write_b32 a216, v15
	v_accvgpr_write_b32 a215, v14
	v_accvgpr_write_b32 a214, v13
	v_accvgpr_write_b32 a213, v12
	s_lshr_b32 s10, s25, 6
	s_add_u32 s8, s8, 0x100
	s_addc_u32 s9, s24, 0
	s_add_u32 s16, s16, 0x100
	s_addc_u32 s17, s7, 0
	s_add_i32 s7, s10, -2
	s_cmp_eq_u32 s10, 2
	v_accvgpr_write_b32 a211, v10
	v_accvgpr_write_b32 a212, v11
	v_accvgpr_write_b32 a222, v8
	v_accvgpr_write_b32 a223, v9
	s_cbranch_scc1 .LBB0_6
; %bb.2:                                ; %.lr.ph.split
	s_mov_b32 s10, 0
	v_mov_b32_e32 v0, 0
	s_mov_b32 s23, 0x27000
	s_mov_b32 s22, 0x7ffffffe
	v_mov_b32_e32 v1, v0
	v_mov_b32_e32 v2, v0
	v_mov_b32_e32 v3, v0
	v_mov_b32_e32 v4, v0
	v_mov_b32_e32 v5, v0
	v_mov_b32_e32 v6, v0
	v_mov_b32_e32 v7, v0
	v_mov_b32_e32 v244, v0
	v_mov_b32_e32 v245, v0
	v_mov_b32_e32 v246, v0
	v_mov_b32_e32 v247, v0
	v_mov_b32_e32 v240, v0
	v_mov_b32_e32 v241, v0
	v_mov_b32_e32 v242, v0
	v_mov_b32_e32 v243, v0
	v_mov_b32_e32 v236, v0
	v_mov_b32_e32 v237, v0
	v_mov_b32_e32 v238, v0
	v_mov_b32_e32 v239, v0
	v_mov_b32_e32 v232, v0
	v_mov_b32_e32 v233, v0
	v_mov_b32_e32 v234, v0
	v_mov_b32_e32 v235, v0
	v_mov_b32_e32 v228, v0
	v_mov_b32_e32 v229, v0
	v_mov_b32_e32 v230, v0
	v_mov_b32_e32 v231, v0
	v_mov_b32_e32 v224, v0
	v_mov_b32_e32 v225, v0
	v_mov_b32_e32 v226, v0
	v_mov_b32_e32 v227, v0
	v_mov_b32_e32 v220, v0
	v_mov_b32_e32 v221, v0
	v_mov_b32_e32 v222, v0
	v_mov_b32_e32 v223, v0
	v_mov_b32_e32 v216, v0
	v_mov_b32_e32 v217, v0
	v_mov_b32_e32 v218, v0
	v_mov_b32_e32 v219, v0
	v_mov_b32_e32 v212, v0
	v_mov_b32_e32 v213, v0
	v_mov_b32_e32 v214, v0
	v_mov_b32_e32 v215, v0
	v_mov_b32_e32 v208, v0
	v_mov_b32_e32 v209, v0
	v_mov_b32_e32 v210, v0
	v_mov_b32_e32 v211, v0
	v_mov_b32_e32 v204, v0
	v_mov_b32_e32 v205, v0
	v_mov_b32_e32 v206, v0
	v_mov_b32_e32 v207, v0
	v_mov_b32_e32 v200, v0
	v_mov_b32_e32 v201, v0
	v_mov_b32_e32 v202, v0
	v_mov_b32_e32 v203, v0
	v_mov_b32_e32 v196, v0
	v_mov_b32_e32 v197, v0
	v_mov_b32_e32 v198, v0
	v_mov_b32_e32 v199, v0
	v_mov_b32_e32 v188, v0
	v_mov_b32_e32 v189, v0
	v_mov_b32_e32 v190, v0
	v_mov_b32_e32 v191, v0
	v_mov_b32_e32 v192, v0
	v_mov_b32_e32 v193, v0
	v_mov_b32_e32 v194, v0
	v_mov_b32_e32 v195, v0
	v_mov_b32_e32 v184, v0
	v_mov_b32_e32 v185, v0
	v_mov_b32_e32 v186, v0
	v_mov_b32_e32 v187, v0
	v_mov_b32_e32 v180, v0
	v_mov_b32_e32 v181, v0
	v_mov_b32_e32 v182, v0
	v_mov_b32_e32 v183, v0
	v_mov_b32_e32 v176, v0
	v_mov_b32_e32 v177, v0
	v_mov_b32_e32 v178, v0
	v_mov_b32_e32 v179, v0
	v_mov_b32_e32 v172, v0
	v_mov_b32_e32 v173, v0
	v_mov_b32_e32 v174, v0
	v_mov_b32_e32 v175, v0
	v_mov_b32_e32 v168, v0
	v_mov_b32_e32 v169, v0
	v_mov_b32_e32 v170, v0
	v_mov_b32_e32 v171, v0
	v_mov_b32_e32 v164, v0
	v_mov_b32_e32 v165, v0
	v_mov_b32_e32 v166, v0
	v_mov_b32_e32 v167, v0
	v_mov_b32_e32 v160, v0
	v_mov_b32_e32 v161, v0
	v_mov_b32_e32 v162, v0
	v_mov_b32_e32 v163, v0
	v_mov_b32_e32 v156, v0
	v_mov_b32_e32 v157, v0
	v_mov_b32_e32 v158, v0
	v_mov_b32_e32 v159, v0
	v_mov_b32_e32 v152, v0
	v_mov_b32_e32 v153, v0
	v_mov_b32_e32 v154, v0
	v_mov_b32_e32 v155, v0
	v_mov_b32_e32 v148, v0
	v_mov_b32_e32 v149, v0
	v_mov_b32_e32 v150, v0
	v_mov_b32_e32 v151, v0
	v_mov_b32_e32 v144, v0
	v_mov_b32_e32 v145, v0
	v_mov_b32_e32 v146, v0
	v_mov_b32_e32 v147, v0
	v_mov_b32_e32 v140, v0
	v_mov_b32_e32 v141, v0
	v_mov_b32_e32 v142, v0
	v_mov_b32_e32 v143, v0
	v_mov_b32_e32 v136, v0
	v_mov_b32_e32 v137, v0
	v_mov_b32_e32 v138, v0
	v_mov_b32_e32 v139, v0
	v_mov_b32_e32 v132, v0
	v_mov_b32_e32 v133, v0
	v_mov_b32_e32 v134, v0
	v_mov_b32_e32 v135, v0
	v_mov_b32_e32 v128, v0
	v_mov_b32_e32 v129, v0
	v_mov_b32_e32 v130, v0
	v_mov_b32_e32 v131, v0
	v_mov_b32_e32 v124, v0
	v_mov_b32_e32 v125, v0
	v_mov_b32_e32 v126, v0
	v_mov_b32_e32 v127, v0
	v_mov_b32_e32 v120, v0
	v_mov_b32_e32 v121, v0
	v_mov_b32_e32 v122, v0
	v_mov_b32_e32 v123, v0
	v_mov_b32_e32 v116, v0
	v_mov_b32_e32 v117, v0
	v_mov_b32_e32 v118, v0
	v_mov_b32_e32 v119, v0
	v_mov_b32_e32 v112, v0
	v_mov_b32_e32 v113, v0
	v_mov_b32_e32 v114, v0
	v_mov_b32_e32 v115, v0
	v_mov_b32_e32 v108, v0
	v_mov_b32_e32 v109, v0
	v_mov_b32_e32 v110, v0
	v_mov_b32_e32 v111, v0
	v_mov_b32_e32 v104, v0
	v_mov_b32_e32 v105, v0
	v_mov_b32_e32 v106, v0
	v_mov_b32_e32 v107, v0
	v_mov_b32_e32 v100, v0
	v_mov_b32_e32 v101, v0
	v_mov_b32_e32 v102, v0
	v_mov_b32_e32 v103, v0
	v_mov_b32_e32 v96, v0
	v_mov_b32_e32 v97, v0
	v_mov_b32_e32 v98, v0
	v_mov_b32_e32 v99, v0
	v_mov_b32_e32 v92, v0
	v_mov_b32_e32 v93, v0
	v_mov_b32_e32 v94, v0
	v_mov_b32_e32 v95, v0
	v_mov_b32_e32 v88, v0
	v_mov_b32_e32 v89, v0
	v_mov_b32_e32 v90, v0
	v_mov_b32_e32 v91, v0
	v_mov_b32_e32 v84, v0
	v_mov_b32_e32 v85, v0
	v_mov_b32_e32 v86, v0
	v_mov_b32_e32 v87, v0
	v_mov_b32_e32 v80, v0
	v_mov_b32_e32 v81, v0
	v_mov_b32_e32 v82, v0
	v_mov_b32_e32 v83, v0
	v_mov_b32_e32 v76, v0
	v_mov_b32_e32 v77, v0
	v_mov_b32_e32 v78, v0
	v_mov_b32_e32 v79, v0
	v_mov_b32_e32 v72, v0
	v_mov_b32_e32 v73, v0
	v_mov_b32_e32 v74, v0
	v_mov_b32_e32 v75, v0
	v_mov_b32_e32 v68, v0
	v_mov_b32_e32 v69, v0
	v_mov_b32_e32 v70, v0
	v_mov_b32_e32 v71, v0
	v_mov_b32_e32 v64, v0
	v_mov_b32_e32 v65, v0
	v_mov_b32_e32 v66, v0
	v_mov_b32_e32 v67, v0
	v_accvgpr_write_b32 a250, v0
	v_accvgpr_write_b32 a251, v0
	v_accvgpr_write_b32 a252, v0
	v_accvgpr_write_b32 a253, v0
	v_mov_b32_e32 v56, v0
	v_mov_b32_e32 v57, v0
	v_mov_b32_e32 v58, v0
	v_mov_b32_e32 v59, v0
	v_mov_b32_e32 v52, v0
	v_mov_b32_e32 v53, v0
	v_mov_b32_e32 v54, v0
	v_mov_b32_e32 v55, v0
	v_mov_b32_e32 v48, v0
	v_mov_b32_e32 v49, v0
	v_mov_b32_e32 v50, v0
	v_mov_b32_e32 v51, v0
	v_mov_b32_e32 v44, v0
	v_mov_b32_e32 v45, v0
	v_mov_b32_e32 v46, v0
	v_mov_b32_e32 v47, v0
	v_mov_b32_e32 v40, v0
	v_mov_b32_e32 v41, v0
	v_mov_b32_e32 v42, v0
	v_mov_b32_e32 v43, v0
	v_mov_b32_e32 v36, v0
	v_mov_b32_e32 v37, v0
	v_mov_b32_e32 v38, v0
	v_mov_b32_e32 v39, v0
	v_mov_b32_e32 v32, v0
	v_mov_b32_e32 v33, v0
	v_mov_b32_e32 v34, v0
	v_mov_b32_e32 v35, v0
	v_mov_b32_e32 v28, v0
	v_mov_b32_e32 v29, v0
	v_mov_b32_e32 v30, v0
	v_mov_b32_e32 v31, v0
	v_mov_b32_e32 v24, v0
	v_mov_b32_e32 v25, v0
	v_mov_b32_e32 v26, v0
	v_mov_b32_e32 v27, v0
	v_mov_b32_e32 v20, v0
	v_mov_b32_e32 v21, v0
	v_mov_b32_e32 v22, v0
	v_mov_b32_e32 v23, v0
	v_mov_b32_e32 v16, v0
	v_mov_b32_e32 v17, v0
	v_mov_b32_e32 v18, v0
	v_mov_b32_e32 v19, v0
	v_mov_b32_e32 v12, v0
	v_mov_b32_e32 v13, v0
	v_mov_b32_e32 v14, v0
	v_mov_b32_e32 v15, v0
	v_mov_b32_e32 v248, v0
	v_mov_b32_e32 v249, v0
	v_mov_b32_e32 v250, v0
	v_mov_b32_e32 v251, v0
	v_accvgpr_write_b32 a228, v248
	v_accvgpr_write_b32 a229, v249
	v_accvgpr_write_b32 a230, v250
	v_accvgpr_write_b32 a231, v251
	v_accvgpr_write_b32 a232, v248
	v_accvgpr_write_b32 a233, v249
	v_accvgpr_write_b32 a234, v250
	v_accvgpr_write_b32 a235, v251
	v_accvgpr_write_b32 a246, v0
	v_accvgpr_write_b32 a247, v0
	v_accvgpr_write_b32 a248, v0
	v_accvgpr_write_b32 a249, v0
	v_accvgpr_write_b32 a226, v254
	v_accvgpr_write_b32 a227, v253
	v_accvgpr_mov_b32 a236, a213
	v_accvgpr_read_b32 v254, a214
	v_accvgpr_mov_b32 a237, a215
	v_accvgpr_read_b32 v253, a216
	v_accvgpr_mov_b32 a238, a217
	v_accvgpr_mov_b32 a242, a218
	v_accvgpr_mov_b32 a239, a219
	v_accvgpr_mov_b32 a243, a220
	v_accvgpr_mov_b32 a240, a221
	v_accvgpr_mov_b32 a244, a224
	v_accvgpr_mov_b32 a241, a225
.LBB0_3:                                ; =>This Inner Loop Header: Depth=1
	s_and_b32 s11, s10, 1
	;;#ASMSTART
	;; Region 0: 128 wmma, 16 GR, 32 LR
	;;#ASMEND
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_16x16x32_f16 v[0:3], a[88:91], a[120:123], v[0:3]
	v_mfma_f32_16x16x32_f16 v[0:3], a[92:95], a[124:127], v[0:3]
	v_mfma_f32_16x16x32_f16 v[4:7], a[80:83], a[120:123], v[4:7]
	v_mfma_f32_16x16x32_f16 v[4:7], a[84:87], a[124:127], v[4:7]
	v_mfma_f32_16x16x32_f16 v[244:247], a[64:67], a[120:123], v[244:247]
	v_mfma_f32_16x16x32_f16 v[244:247], a[68:71], a[124:127], v[244:247]
	v_mfma_f32_16x16x32_f16 v[240:243], a[56:59], a[120:123], v[240:243]
	v_mfma_f32_16x16x32_f16 v[240:243], a[60:63], a[124:127], v[240:243]
	v_mfma_f32_16x16x32_f16 v[236:239], a[48:51], a[120:123], v[236:239]
	v_mfma_f32_16x16x32_f16 v[236:239], a[52:55], a[124:127], v[236:239]
	v_mfma_f32_16x16x32_f16 v[232:235], a[40:43], a[120:123], v[232:235]
	v_mfma_f32_16x16x32_f16 v[232:235], a[44:47], a[124:127], v[232:235]
	v_mfma_f32_16x16x32_f16 v[228:231], a[32:35], a[120:123], v[228:231]
	v_mfma_f32_16x16x32_f16 v[228:231], a[36:39], a[124:127], v[228:231]
	v_mfma_f32_16x16x32_f16 v[224:227], a[24:27], a[120:123], v[224:227]
	v_mfma_f32_16x16x32_f16 v[224:227], a[28:31], a[124:127], v[224:227]
	v_mfma_f32_16x16x32_f16 v[220:223], a[88:91], a[112:115], v[220:223]
	v_mfma_f32_16x16x32_f16 v[220:223], a[92:95], a[116:119], v[220:223]
	v_mfma_f32_16x16x32_f16 v[216:219], a[80:83], a[112:115], v[216:219]
	v_mfma_f32_16x16x32_f16 v[216:219], a[84:87], a[116:119], v[216:219]
	v_mfma_f32_16x16x32_f16 v[212:215], a[64:67], a[112:115], v[212:215]
	v_mfma_f32_16x16x32_f16 v[212:215], a[68:71], a[116:119], v[212:215]
	v_mfma_f32_16x16x32_f16 v[208:211], a[56:59], a[112:115], v[208:211]
	v_mfma_f32_16x16x32_f16 v[208:211], a[60:63], a[116:119], v[208:211]
	v_mfma_f32_16x16x32_f16 v[204:207], a[48:51], a[112:115], v[204:207]
	v_mfma_f32_16x16x32_f16 v[204:207], a[52:55], a[116:119], v[204:207]
	v_mfma_f32_16x16x32_f16 v[200:203], a[40:43], a[112:115], v[200:203]
	v_mfma_f32_16x16x32_f16 v[200:203], a[44:47], a[116:119], v[200:203]
	v_mfma_f32_16x16x32_f16 v[196:199], a[32:35], a[112:115], v[196:199]
	v_mfma_f32_16x16x32_f16 v[196:199], a[36:39], a[116:119], v[196:199]
	v_mfma_f32_16x16x32_f16 v[188:191], a[24:27], a[112:115], v[188:191]
	v_mfma_f32_16x16x32_f16 v[188:191], a[28:31], a[116:119], v[188:191]
	v_mfma_f32_16x16x32_f16 v[192:195], a[88:91], a[104:107], v[192:195]
	; wait_asyncmark(0)
	s_waitcnt vmcnt(0)
	s_barrier
	s_mul_i32 s15, s11, 0x8400
	s_add_i32 s15, s15, 0
	s_and_b32 s21, s17, 0xffff
	s_mov_b32 s20, s16
	s_add_i32 s15, s15, s6
	s_mov_b32 m0, s15
	s_nop 0
	buffer_load_dwordx4 v252, s[20:23], 0 offen lds
	v_mfma_f32_16x16x32_f16 v[192:195], a[92:95], a[108:111], v[192:195]
	v_mfma_f32_16x16x32_f16 v[184:187], a[80:83], a[104:107], v[184:187]
	v_mfma_f32_16x16x32_f16 v[184:187], a[84:87], a[108:111], v[184:187]
	v_mfma_f32_16x16x32_f16 v[180:183], a[64:67], a[104:107], v[180:183]
	s_add_i32 m0, s15, 0x1080
	s_nop 0
	buffer_load_dwordx4 v10, s[20:23], 0 offen lds
	v_mfma_f32_16x16x32_f16 v[180:183], a[68:71], a[108:111], v[180:183]
	v_mfma_f32_16x16x32_f16 v[176:179], a[56:59], a[104:107], v[176:179]
	v_mfma_f32_16x16x32_f16 v[176:179], a[60:63], a[108:111], v[176:179]
	v_mfma_f32_16x16x32_f16 v[172:175], a[48:51], a[104:107], v[172:175]
	s_add_i32 m0, s15, 0x2100
	s_nop 0
	buffer_load_dwordx4 v11, s[20:23], 0 offen lds
	v_mfma_f32_16x16x32_f16 v[172:175], a[52:55], a[108:111], v[172:175]
	v_mfma_f32_16x16x32_f16 v[168:171], a[40:43], a[104:107], v[168:171]
	v_mfma_f32_16x16x32_f16 v[168:171], a[44:47], a[108:111], v[168:171]
	v_mfma_f32_16x16x32_f16 v[164:167], a[32:35], a[104:107], v[164:167]
	s_add_i32 m0, s15, 0x3180
	v_accvgpr_read_b32 v10, a236
	buffer_load_dwordx4 v10, s[20:23], 0 offen lds
	v_mfma_f32_16x16x32_f16 v[164:167], a[36:39], a[108:111], v[164:167]
	v_mfma_f32_16x16x32_f16 v[160:163], a[24:27], a[104:107], v[160:163]
	v_mfma_f32_16x16x32_f16 v[160:163], a[28:31], a[108:111], v[160:163]
	v_mfma_f32_16x16x32_f16 v[156:159], a[88:91], a[96:99], v[156:159]
	s_add_i32 m0, s15, 0x4200
	v_accvgpr_read_b32 v10, a227
	buffer_load_dwordx4 v10, s[20:23], 0 offen lds
	v_mfma_f32_16x16x32_f16 v[156:159], a[92:95], a[100:103], v[156:159]
	v_mfma_f32_16x16x32_f16 v[152:155], a[80:83], a[96:99], v[152:155]
	v_mfma_f32_16x16x32_f16 v[152:155], a[84:87], a[100:103], v[152:155]
	v_mfma_f32_16x16x32_f16 v[148:151], a[64:67], a[96:99], v[148:151]
	s_add_i32 m0, s15, 0x5280
	v_accvgpr_read_b32 v10, a226
	buffer_load_dwordx4 v10, s[20:23], 0 offen lds
	v_mfma_f32_16x16x32_f16 v[148:151], a[68:71], a[100:103], v[148:151]
	v_mfma_f32_16x16x32_f16 v[144:147], a[56:59], a[96:99], v[144:147]
	v_mfma_f32_16x16x32_f16 v[144:147], a[60:63], a[100:103], v[144:147]
	v_mfma_f32_16x16x32_f16 v[140:143], a[48:51], a[96:99], v[140:143]
	s_add_i32 m0, s15, 0x6300
	s_nop 0
	buffer_load_dwordx4 v254, s[20:23], 0 offen lds
	v_mfma_f32_16x16x32_f16 v[140:143], a[52:55], a[100:103], v[140:143]
	v_mfma_f32_16x16x32_f16 v[136:139], a[40:43], a[96:99], v[136:139]
	v_mfma_f32_16x16x32_f16 v[136:139], a[44:47], a[100:103], v[136:139]
	v_mfma_f32_16x16x32_f16 v[132:135], a[32:35], a[96:99], v[132:135]
	s_add_i32 m0, s15, 0x7380
	v_accvgpr_read_b32 v10, a237
	buffer_load_dwordx4 v10, s[20:23], 0 offen lds
	v_mfma_f32_16x16x32_f16 v[132:135], a[36:39], a[100:103], v[132:135]
	v_mfma_f32_16x16x32_f16 v[128:131], a[24:27], a[96:99], v[128:131]
	v_mfma_f32_16x16x32_f16 v[128:131], a[28:31], a[100:103], v[128:131]
	v_mfma_f32_16x16x32_f16 v[124:127], a[88:91], a[16:19], v[124:127]
	s_and_b32 s21, s9, 0xffff
	s_mov_b32 s20, s8
	s_add_i32 m0, s15, 0x107e0
	s_nop 0
	buffer_load_dwordx4 v253, s[20:23], 0 offen lds
	v_mfma_f32_16x16x32_f16 v[124:127], a[92:95], a[20:23], v[124:127]
	v_mfma_f32_16x16x32_f16 v[120:123], a[80:83], a[16:19], v[120:123]
	v_mfma_f32_16x16x32_f16 v[120:123], a[84:87], a[20:23], v[120:123]
	v_mfma_f32_16x16x32_f16 v[116:119], a[64:67], a[16:19], v[116:119]
	s_add_i32 m0, s15, 0x11860
	v_accvgpr_read_b32 v10, a238
	buffer_load_dwordx4 v10, s[20:23], 0 offen lds
	v_mfma_f32_16x16x32_f16 v[116:119], a[68:71], a[20:23], v[116:119]
	v_mfma_f32_16x16x32_f16 v[112:115], a[56:59], a[16:19], v[112:115]
	v_mfma_f32_16x16x32_f16 v[112:115], a[60:63], a[20:23], v[112:115]
	v_mfma_f32_16x16x32_f16 v[108:111], a[48:51], a[16:19], v[108:111]
	s_add_i32 m0, s15, 0x128e0
	v_accvgpr_read_b32 v10, a242
	buffer_load_dwordx4 v10, s[20:23], 0 offen lds
	v_mfma_f32_16x16x32_f16 v[108:111], a[52:55], a[20:23], v[108:111]
	v_mfma_f32_16x16x32_f16 v[104:107], a[40:43], a[16:19], v[104:107]
	v_mfma_f32_16x16x32_f16 v[104:107], a[44:47], a[20:23], v[104:107]
	v_mfma_f32_16x16x32_f16 v[100:103], a[32:35], a[16:19], v[100:103]
	s_add_i32 m0, s15, 0x13960
	v_accvgpr_read_b32 v10, a239
	buffer_load_dwordx4 v10, s[20:23], 0 offen lds
	v_mfma_f32_16x16x32_f16 v[100:103], a[36:39], a[20:23], v[100:103]
	v_mfma_f32_16x16x32_f16 v[96:99], a[24:27], a[16:19], v[96:99]
	v_mfma_f32_16x16x32_f16 v[96:99], a[28:31], a[20:23], v[96:99]
	v_mfma_f32_16x16x32_f16 v[92:95], a[88:91], a[0:3], v[92:95]
	s_add_i32 m0, s15, 0x149e0
	v_accvgpr_read_b32 v10, a243
	buffer_load_dwordx4 v10, s[20:23], 0 offen lds
	v_mfma_f32_16x16x32_f16 v[92:95], a[92:95], a[4:7], v[92:95]
	v_mfma_f32_16x16x32_f16 v[88:91], a[80:83], a[0:3], v[88:91]
	v_mfma_f32_16x16x32_f16 v[88:91], a[84:87], a[4:7], v[88:91]
	v_mfma_f32_16x16x32_f16 v[84:87], a[64:67], a[0:3], v[84:87]
	s_add_i32 m0, s15, 0x15a60
	v_accvgpr_read_b32 v10, a240
	buffer_load_dwordx4 v10, s[20:23], 0 offen lds
	v_mfma_f32_16x16x32_f16 v[84:87], a[68:71], a[4:7], v[84:87]
	v_mfma_f32_16x16x32_f16 v[80:83], a[56:59], a[0:3], v[80:83]
	v_mfma_f32_16x16x32_f16 v[80:83], a[60:63], a[4:7], v[80:83]
	v_mfma_f32_16x16x32_f16 v[76:79], a[48:51], a[0:3], v[76:79]
	s_add_i32 m0, s15, 0x16ae0
	s_nop 0
	buffer_load_dwordx4 v8, s[20:23], 0 offen lds
	v_mfma_f32_16x16x32_f16 v[76:79], a[52:55], a[4:7], v[76:79]
	v_mfma_f32_16x16x32_f16 v[72:75], a[40:43], a[0:3], v[72:75]
	v_mfma_f32_16x16x32_f16 v[72:75], a[44:47], a[4:7], v[72:75]
	v_mfma_f32_16x16x32_f16 v[68:71], a[32:35], a[0:3], v[68:71]
	s_add_i32 m0, s15, 0x17b60
	s_nop 0
	buffer_load_dwordx4 v9, s[20:23], 0 offen lds
	v_mfma_f32_16x16x32_f16 v[68:71], a[36:39], a[4:7], v[68:71]
	; asyncmark
	s_lshl_b32 s11, s11, 14
	s_xor_b32 s11, s11, 0x4000
	s_lshr_b32 s15, s11, 5
	s_or_b32 s15, s15, s11
	s_lshl1_add_u32 s11, s15, 0
	s_barrier
	v_accvgpr_read_b32 v8, a244
	v_add_u32_e32 v8, s11, v8
	ds_read_b128 a[120:123], v8
	v_mfma_f32_16x16x32_f16 v[64:67], a[24:27], a[0:3], v[64:67]
	ds_read_b128 a[124:127], v8 offset:64
	v_mfma_f32_16x16x32_f16 v[64:67], a[28:31], a[4:7], v[64:67]
	ds_read_b128 a[112:115], v8 offset:256
	v_mfma_f32_16x16x32_f16 a[136:139], a[88:91], a[72:75], a[250:253]
	ds_read_b128 a[116:119], v8 offset:320
	v_mfma_f32_16x16x32_f16 a[250:253], a[92:95], a[76:79], a[136:139]
	ds_read_b128 a[104:107], v8 offset:512
	v_mfma_f32_16x16x32_f16 v[56:59], a[80:83], a[72:75], v[56:59]
	ds_read_b128 a[108:111], v8 offset:576
	v_mfma_f32_16x16x32_f16 v[56:59], a[84:87], a[76:79], v[56:59]
	ds_read_b128 a[96:99], v8 offset:768
	v_mfma_f32_16x16x32_f16 v[52:55], a[64:67], a[72:75], v[52:55]
	ds_read_b128 a[100:103], v8 offset:832
	v_mfma_f32_16x16x32_f16 v[52:55], a[68:71], a[76:79], v[52:55]
	ds_read_b128 a[16:19], v8 offset:16896
	v_mfma_f32_16x16x32_f16 v[48:51], a[56:59], a[72:75], v[48:51]
	ds_read_b128 a[20:23], v8 offset:16960
	v_mfma_f32_16x16x32_f16 v[48:51], a[60:63], a[76:79], v[48:51]
	ds_read_b128 a[0:3], v8 offset:17152
	v_mfma_f32_16x16x32_f16 v[44:47], a[48:51], a[72:75], v[44:47]
	ds_read_b128 a[4:7], v8 offset:17216
	v_mfma_f32_16x16x32_f16 v[44:47], a[52:55], a[76:79], v[44:47]
	ds_read_b128 a[128:131], v8 offset:17408
	v_mfma_f32_16x16x32_f16 v[40:43], a[40:43], a[72:75], v[40:43]
	ds_read_b128 a[132:135], v8 offset:17472
	v_mfma_f32_16x16x32_f16 v[40:43], a[44:47], a[76:79], v[40:43]
	ds_read_b128 a[136:139], v8 offset:17664
	v_mfma_f32_16x16x32_f16 v[36:39], a[32:35], a[72:75], v[36:39]
	ds_read_b128 a[140:143], v8 offset:17728
	v_mfma_f32_16x16x32_f16 v[36:39], a[36:39], a[76:79], v[36:39]
	v_accvgpr_read_b32 v8, a241
	v_add_u32_e32 v8, s11, v8
	v_accvgpr_write_b32 a210, v252
	v_add_u32_e32 v252, 0x107e0, v8
	ds_read_b128 a[144:147], v252
	v_mfma_f32_16x16x32_f16 v[32:35], a[24:27], a[72:75], v[32:35]
	ds_read_b128 a[148:151], v252 offset:64
	v_mfma_f32_16x16x32_f16 v[32:35], a[28:31], a[76:79], v[32:35]
	ds_read_b128 a[152:155], v252 offset:256
	v_mfma_f32_16x16x32_f16 v[28:31], a[88:91], a[8:11], v[28:31]
	ds_read_b128 a[156:159], v252 offset:320
	v_mfma_f32_16x16x32_f16 v[28:31], a[92:95], a[12:15], v[28:31]
	ds_read_b128 a[160:163], v252 offset:512
	v_mfma_f32_16x16x32_f16 v[24:27], a[80:83], a[8:11], v[24:27]
	ds_read_b128 a[164:167], v252 offset:576
	v_mfma_f32_16x16x32_f16 v[24:27], a[84:87], a[12:15], v[24:27]
	ds_read_b128 a[168:171], v252 offset:768
	v_mfma_f32_16x16x32_f16 v[20:23], a[64:67], a[8:11], v[20:23]
	ds_read_b128 a[172:175], v252 offset:832
	v_mfma_f32_16x16x32_f16 v[20:23], a[68:71], a[12:15], v[20:23]
	ds_read_b128 a[176:179], v252 offset:16896
	v_mfma_f32_16x16x32_f16 v[16:19], a[56:59], a[8:11], v[16:19]
	ds_read_b128 a[180:183], v252 offset:16960
	v_mfma_f32_16x16x32_f16 v[16:19], a[60:63], a[12:15], v[16:19]
	ds_read_b128 a[184:187], v252 offset:17152
	v_mfma_f32_16x16x32_f16 v[12:15], a[48:51], a[8:11], v[12:15]
	ds_read_b128 a[188:191], v252 offset:17216
	v_mfma_f32_16x16x32_f16 v[12:15], a[52:55], a[12:15], v[12:15]
	ds_read_b128 a[192:195], v252 offset:17408
	v_mfma_f32_16x16x32_f16 a[228:231], a[40:43], a[8:11], a[228:231]
	ds_read_b128 a[196:199], v252 offset:17472
	v_mfma_f32_16x16x32_f16 a[228:231], a[44:47], a[12:15], a[228:231]
	ds_read_b128 a[200:203], v252 offset:17664
	v_mfma_f32_16x16x32_f16 a[232:235], a[32:35], a[8:11], a[232:235]
	ds_read_b128 a[204:207], v252 offset:17728
	v_accvgpr_read_b32 v252, a210
	v_mfma_f32_16x16x32_f16 a[232:235], a[36:39], a[12:15], a[232:235]
	v_mfma_f32_16x16x32_f16 a[8:11], a[24:27], a[8:11], a[246:249]
	v_mfma_f32_16x16x32_f16 a[246:249], a[28:31], a[12:15], a[8:11]
	v_accvgpr_read_b32 v9, a223
	v_accvgpr_read_b32 v8, a222
	v_accvgpr_read_b32 v11, a212
	v_accvgpr_read_b32 v10, a211
	s_add_u32 s16, s16, 0x80
	s_addc_u32 s17, s17, 0
	s_add_u32 s8, s8, 0x80
	s_addc_u32 s9, s9, 0
	s_add_i32 s10, s10, 1
	s_waitcnt lgkmcnt(14)
	v_accvgpr_mov_b32 a72, a128
	v_accvgpr_mov_b32 a73, a129
	v_accvgpr_mov_b32 a74, a130
	v_accvgpr_mov_b32 a75, a131
	v_accvgpr_mov_b32 a76, a132
	v_accvgpr_mov_b32 a77, a133
	v_accvgpr_mov_b32 a78, a134
	v_accvgpr_mov_b32 a79, a135
	v_accvgpr_mov_b32 a8, a136
	v_accvgpr_mov_b32 a9, a137
	v_accvgpr_mov_b32 a10, a138
	v_accvgpr_mov_b32 a11, a139
	v_accvgpr_mov_b32 a12, a140
	v_accvgpr_mov_b32 a13, a141
	v_accvgpr_mov_b32 a14, a142
	v_accvgpr_mov_b32 a15, a143
	v_accvgpr_mov_b32 a88, a144
	v_accvgpr_mov_b32 a89, a145
	v_accvgpr_mov_b32 a90, a146
	v_accvgpr_mov_b32 a91, a147
	v_accvgpr_mov_b32 a92, a148
	v_accvgpr_mov_b32 a93, a149
	v_accvgpr_mov_b32 a94, a150
	v_accvgpr_mov_b32 a95, a151
	s_waitcnt lgkmcnt(13)
	v_accvgpr_mov_b32 a80, a152
	v_accvgpr_mov_b32 a81, a153
	v_accvgpr_mov_b32 a82, a154
	v_accvgpr_mov_b32 a83, a155
	s_waitcnt lgkmcnt(12)
	v_accvgpr_mov_b32 a84, a156
	v_accvgpr_mov_b32 a85, a157
	v_accvgpr_mov_b32 a86, a158
	v_accvgpr_mov_b32 a87, a159
	s_waitcnt lgkmcnt(11)
	v_accvgpr_mov_b32 a64, a160
	v_accvgpr_mov_b32 a65, a161
	v_accvgpr_mov_b32 a66, a162
	v_accvgpr_mov_b32 a67, a163
	s_waitcnt lgkmcnt(10)
	v_accvgpr_mov_b32 a68, a164
	v_accvgpr_mov_b32 a69, a165
	v_accvgpr_mov_b32 a70, a166
	v_accvgpr_mov_b32 a71, a167
	s_waitcnt lgkmcnt(9)
	v_accvgpr_mov_b32 a56, a168
	v_accvgpr_mov_b32 a57, a169
	v_accvgpr_mov_b32 a58, a170
	v_accvgpr_mov_b32 a59, a171
	s_waitcnt lgkmcnt(8)
	v_accvgpr_mov_b32 a60, a172
	v_accvgpr_mov_b32 a61, a173
	v_accvgpr_mov_b32 a62, a174
	v_accvgpr_mov_b32 a63, a175
	s_waitcnt lgkmcnt(7)
	v_accvgpr_mov_b32 a48, a176
	v_accvgpr_mov_b32 a49, a177
	v_accvgpr_mov_b32 a50, a178
	v_accvgpr_mov_b32 a51, a179
	s_waitcnt lgkmcnt(6)
	v_accvgpr_mov_b32 a52, a180
	v_accvgpr_mov_b32 a53, a181
	v_accvgpr_mov_b32 a54, a182
	v_accvgpr_mov_b32 a55, a183
	s_waitcnt lgkmcnt(5)
	v_accvgpr_mov_b32 a40, a184
	v_accvgpr_mov_b32 a41, a185
	v_accvgpr_mov_b32 a42, a186
	v_accvgpr_mov_b32 a43, a187
	s_waitcnt lgkmcnt(4)
	v_accvgpr_mov_b32 a44, a188
	v_accvgpr_mov_b32 a45, a189
	v_accvgpr_mov_b32 a46, a190
	v_accvgpr_mov_b32 a47, a191
	s_waitcnt lgkmcnt(3)
	v_accvgpr_mov_b32 a32, a192
	v_accvgpr_mov_b32 a33, a193
	v_accvgpr_mov_b32 a34, a194
	v_accvgpr_mov_b32 a35, a195
	s_waitcnt lgkmcnt(2)
	v_accvgpr_mov_b32 a36, a196
	v_accvgpr_mov_b32 a37, a197
	v_accvgpr_mov_b32 a38, a198
	v_accvgpr_mov_b32 a39, a199
	s_waitcnt lgkmcnt(1)
	v_accvgpr_mov_b32 a24, a200
	v_accvgpr_mov_b32 a25, a201
	v_accvgpr_mov_b32 a26, a202
	v_accvgpr_mov_b32 a27, a203
	s_waitcnt lgkmcnt(0)
	v_accvgpr_mov_b32 a28, a204
	v_accvgpr_mov_b32 a29, a205
	v_accvgpr_mov_b32 a30, a206
	v_accvgpr_mov_b32 a31, a207
	s_cmp_lg_u32 s7, s10
	s_cbranch_scc1 .LBB0_3
; %bb.4:                                ; %Flow
	v_accvgpr_mov_b32 a132, a228
	v_accvgpr_mov_b32 a133, a229
	v_accvgpr_mov_b32 a134, a230
	v_accvgpr_mov_b32 a135, a231
	v_accvgpr_mov_b32 a128, a232
	v_accvgpr_mov_b32 a129, a233
	v_accvgpr_mov_b32 a130, a234
	v_accvgpr_mov_b32 a131, a235
	s_mov_b32 s15, s7
	v_accvgpr_read_b32 v253, a227
	v_accvgpr_read_b32 v254, a226
	s_branch .LBB0_7
.LBB0_5:
	v_accvgpr_write_b32 a131, 0
	v_accvgpr_write_b32 a130, 0
	v_accvgpr_write_b32 a129, 0
	v_accvgpr_write_b32 a128, 0
	v_accvgpr_write_b32 a191, 0
	v_accvgpr_write_b32 a190, 0
	v_accvgpr_write_b32 a189, 0
	v_accvgpr_write_b32 a188, 0
	v_accvgpr_write_b32 a135, 0
	v_accvgpr_write_b32 a134, 0
	v_accvgpr_write_b32 a133, 0
	v_accvgpr_write_b32 a132, 0
	v_accvgpr_write_b32 a187, 0
	v_accvgpr_write_b32 a186, 0
	v_accvgpr_write_b32 a185, 0
	v_accvgpr_write_b32 a184, 0
	v_accvgpr_write_b32 a183, 0
	v_accvgpr_write_b32 a182, 0
	v_accvgpr_write_b32 a181, 0
	v_accvgpr_write_b32 a180, 0
	v_accvgpr_write_b32 a179, 0
	v_accvgpr_write_b32 a178, 0
	v_accvgpr_write_b32 a177, 0
	v_accvgpr_write_b32 a176, 0
	v_accvgpr_write_b32 a175, 0
	v_accvgpr_write_b32 a174, 0
	v_accvgpr_write_b32 a173, 0
	v_accvgpr_write_b32 a172, 0
	v_accvgpr_write_b32 a171, 0
	v_accvgpr_write_b32 a170, 0
	v_accvgpr_write_b32 a169, 0
	v_accvgpr_write_b32 a168, 0
	v_accvgpr_write_b32 a167, 0
	v_accvgpr_write_b32 a166, 0
	v_accvgpr_write_b32 a165, 0
	v_accvgpr_write_b32 a164, 0
	v_accvgpr_write_b32 a163, 0
	v_accvgpr_write_b32 a162, 0
	v_accvgpr_write_b32 a161, 0
	v_accvgpr_write_b32 a160, 0
	v_accvgpr_write_b32 a159, 0
	v_accvgpr_write_b32 a158, 0
	v_accvgpr_write_b32 a157, 0
	v_accvgpr_write_b32 a156, 0
	v_accvgpr_write_b32 a155, 0
	v_accvgpr_write_b32 a154, 0
	v_accvgpr_write_b32 a153, 0
	v_accvgpr_write_b32 a152, 0
	v_accvgpr_write_b32 a151, 0
	v_accvgpr_write_b32 a150, 0
	v_accvgpr_write_b32 a149, 0
	v_accvgpr_write_b32 a148, 0
	v_accvgpr_write_b32 a147, 0
	v_accvgpr_write_b32 a146, 0
	v_accvgpr_write_b32 a145, 0
	v_accvgpr_write_b32 a144, 0
	v_accvgpr_write_b32 a143, 0
	v_accvgpr_write_b32 a142, 0
	v_accvgpr_write_b32 a141, 0
	v_accvgpr_write_b32 a140, 0
	v_accvgpr_write_b32 a139, 0
	v_accvgpr_write_b32 a138, 0
	v_accvgpr_write_b32 a137, 0
	v_accvgpr_write_b32 a136, 0
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
	v_mov_b32_e32 v99, 0
	v_mov_b32_e32 v98, 0
	v_mov_b32_e32 v97, 0
	v_mov_b32_e32 v96, 0
	v_mov_b32_e32 v103, 0
	v_mov_b32_e32 v102, 0
	v_mov_b32_e32 v101, 0
	v_mov_b32_e32 v100, 0
	v_mov_b32_e32 v107, 0
	v_mov_b32_e32 v106, 0
	v_mov_b32_e32 v105, 0
	v_mov_b32_e32 v104, 0
	v_mov_b32_e32 v111, 0
	v_mov_b32_e32 v110, 0
	v_mov_b32_e32 v109, 0
	v_mov_b32_e32 v108, 0
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
	v_mov_b32_e32 v131, 0
	v_mov_b32_e32 v130, 0
	v_mov_b32_e32 v129, 0
	v_mov_b32_e32 v128, 0
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
	v_mov_b32_e32 v151, 0
	v_mov_b32_e32 v150, 0
	v_mov_b32_e32 v149, 0
	v_mov_b32_e32 v148, 0
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
	v_mov_b32_e32 v195, 0
	v_mov_b32_e32 v194, 0
	v_mov_b32_e32 v193, 0
	v_mov_b32_e32 v192, 0
	v_mov_b32_e32 v191, 0
	v_mov_b32_e32 v190, 0
	v_mov_b32_e32 v189, 0
	v_mov_b32_e32 v188, 0
	v_mov_b32_e32 v199, 0
	v_mov_b32_e32 v198, 0
	v_mov_b32_e32 v197, 0
	v_mov_b32_e32 v196, 0
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
	v_mov_b32_e32 v227, 0
	v_mov_b32_e32 v226, 0
	v_mov_b32_e32 v225, 0
	v_mov_b32_e32 v224, 0
	v_mov_b32_e32 v231, 0
	v_mov_b32_e32 v230, 0
	v_mov_b32_e32 v229, 0
	v_mov_b32_e32 v228, 0
	v_mov_b32_e32 v235, 0
	v_mov_b32_e32 v234, 0
	v_mov_b32_e32 v233, 0
	v_mov_b32_e32 v232, 0
	v_mov_b32_e32 v239, 0
	v_mov_b32_e32 v238, 0
	v_mov_b32_e32 v237, 0
	v_mov_b32_e32 v236, 0
	v_mov_b32_e32 v243, 0
	v_mov_b32_e32 v242, 0
	v_mov_b32_e32 v241, 0
	v_mov_b32_e32 v240, 0
	v_mov_b32_e32 v247, 0
	v_mov_b32_e32 v246, 0
	v_mov_b32_e32 v245, 0
	v_mov_b32_e32 v244, 0
	v_mov_b32_e32 v7, 0
	v_mov_b32_e32 v6, 0
	v_mov_b32_e32 v5, 0
	v_mov_b32_e32 v4, 0
	v_accvgpr_write_b32 a195, 0
	v_accvgpr_write_b32 a194, 0
	v_accvgpr_write_b32 a193, 0
	v_accvgpr_write_b32 a192, 0
	s_branch .LBB0_8
.LBB0_6:
	v_accvgpr_write_b32 a249, 0
	v_accvgpr_mov_b32 a248, a249
	v_accvgpr_mov_b32 a247, a249
	v_accvgpr_mov_b32 a246, a249
	v_accvgpr_read_b32 v3, a249
	v_accvgpr_read_b32 v2, a249
	v_accvgpr_read_b32 v1, a249
	v_accvgpr_read_b32 v0, a249
	v_accvgpr_write_b32 a131, v3
	v_accvgpr_write_b32 a130, v2
	v_accvgpr_write_b32 a129, v1
	v_accvgpr_write_b32 a128, v0
	v_accvgpr_write_b32 a135, v3
	v_accvgpr_write_b32 a134, v2
	v_accvgpr_write_b32 a133, v1
	v_accvgpr_write_b32 a132, v0
	v_accvgpr_read_b32 v15, a249
	v_accvgpr_read_b32 v14, a249
	v_accvgpr_read_b32 v13, a249
	v_accvgpr_read_b32 v12, a249
	v_accvgpr_read_b32 v19, a249
	v_accvgpr_read_b32 v18, a249
	v_accvgpr_read_b32 v17, a249
	v_accvgpr_read_b32 v16, a249
	v_accvgpr_read_b32 v23, a249
	v_accvgpr_read_b32 v22, a249
	v_accvgpr_read_b32 v21, a249
	v_accvgpr_read_b32 v20, a249
	v_accvgpr_read_b32 v27, a249
	v_accvgpr_read_b32 v26, a249
	v_accvgpr_read_b32 v25, a249
	v_accvgpr_read_b32 v24, a249
	v_accvgpr_read_b32 v31, a249
	v_accvgpr_read_b32 v30, a249
	v_accvgpr_read_b32 v29, a249
	v_accvgpr_read_b32 v28, a249
	v_accvgpr_read_b32 v35, a249
	v_accvgpr_read_b32 v34, a249
	v_accvgpr_read_b32 v33, a249
	v_accvgpr_read_b32 v32, a249
	v_accvgpr_read_b32 v39, a249
	v_accvgpr_read_b32 v38, a249
	v_accvgpr_read_b32 v37, a249
	v_accvgpr_read_b32 v36, a249
	v_accvgpr_read_b32 v43, a249
	v_accvgpr_read_b32 v42, a249
	v_accvgpr_read_b32 v41, a249
	v_accvgpr_read_b32 v40, a249
	v_accvgpr_read_b32 v47, a249
	v_accvgpr_read_b32 v46, a249
	v_accvgpr_read_b32 v45, a249
	v_accvgpr_read_b32 v44, a249
	v_accvgpr_read_b32 v51, a249
	v_accvgpr_read_b32 v50, a249
	v_accvgpr_read_b32 v49, a249
	v_accvgpr_read_b32 v48, a249
	v_accvgpr_read_b32 v55, a249
	v_accvgpr_read_b32 v54, a249
	v_accvgpr_read_b32 v53, a249
	v_accvgpr_read_b32 v52, a249
	v_accvgpr_read_b32 v59, a249
	v_accvgpr_read_b32 v58, a249
	v_accvgpr_read_b32 v57, a249
	v_accvgpr_read_b32 v56, a249
	v_accvgpr_mov_b32 a253, a249
	v_accvgpr_mov_b32 a252, a249
	v_accvgpr_mov_b32 a251, a249
	v_accvgpr_mov_b32 a250, a249
	v_accvgpr_read_b32 v67, a249
	v_accvgpr_read_b32 v66, a249
	v_accvgpr_read_b32 v65, a249
	v_accvgpr_read_b32 v64, a249
	v_accvgpr_read_b32 v71, a249
	v_accvgpr_read_b32 v70, a249
	v_accvgpr_read_b32 v69, a249
	v_accvgpr_read_b32 v68, a249
	v_accvgpr_read_b32 v75, a249
	v_accvgpr_read_b32 v74, a249
	v_accvgpr_read_b32 v73, a249
	v_accvgpr_read_b32 v72, a249
	v_accvgpr_read_b32 v79, a249
	v_accvgpr_read_b32 v78, a249
	v_accvgpr_read_b32 v77, a249
	v_accvgpr_read_b32 v76, a249
	v_accvgpr_read_b32 v83, a249
	v_accvgpr_read_b32 v82, a249
	v_accvgpr_read_b32 v81, a249
	v_accvgpr_read_b32 v80, a249
	v_accvgpr_read_b32 v87, a249
	v_accvgpr_read_b32 v86, a249
	v_accvgpr_read_b32 v85, a249
	v_accvgpr_read_b32 v84, a249
	v_accvgpr_read_b32 v91, a249
	v_accvgpr_read_b32 v90, a249
	v_accvgpr_read_b32 v89, a249
	v_accvgpr_read_b32 v88, a249
	v_accvgpr_read_b32 v95, a249
	v_accvgpr_read_b32 v94, a249
	v_accvgpr_read_b32 v93, a249
	v_accvgpr_read_b32 v92, a249
	v_accvgpr_read_b32 v99, a249
	v_accvgpr_read_b32 v98, a249
	v_accvgpr_read_b32 v97, a249
	v_accvgpr_read_b32 v96, a249
	v_accvgpr_read_b32 v103, a249
	v_accvgpr_read_b32 v102, a249
	v_accvgpr_read_b32 v101, a249
	v_accvgpr_read_b32 v100, a249
	v_accvgpr_read_b32 v107, a249
	v_accvgpr_read_b32 v106, a249
	v_accvgpr_read_b32 v105, a249
	v_accvgpr_read_b32 v104, a249
	v_accvgpr_read_b32 v111, a249
	v_accvgpr_read_b32 v110, a249
	v_accvgpr_read_b32 v109, a249
	v_accvgpr_read_b32 v108, a249
	v_accvgpr_read_b32 v115, a249
	v_accvgpr_read_b32 v114, a249
	v_accvgpr_read_b32 v113, a249
	v_accvgpr_read_b32 v112, a249
	v_accvgpr_read_b32 v119, a249
	v_accvgpr_read_b32 v118, a249
	v_accvgpr_read_b32 v117, a249
	v_accvgpr_read_b32 v116, a249
	v_accvgpr_read_b32 v123, a249
	v_accvgpr_read_b32 v122, a249
	v_accvgpr_read_b32 v121, a249
	v_accvgpr_read_b32 v120, a249
	v_accvgpr_read_b32 v127, a249
	v_accvgpr_read_b32 v126, a249
	v_accvgpr_read_b32 v125, a249
	v_accvgpr_read_b32 v124, a249
	v_accvgpr_read_b32 v131, a249
	v_accvgpr_read_b32 v130, a249
	v_accvgpr_read_b32 v129, a249
	v_accvgpr_read_b32 v128, a249
	v_accvgpr_read_b32 v135, a249
	v_accvgpr_read_b32 v134, a249
	v_accvgpr_read_b32 v133, a249
	v_accvgpr_read_b32 v132, a249
	v_accvgpr_read_b32 v139, a249
	v_accvgpr_read_b32 v138, a249
	v_accvgpr_read_b32 v137, a249
	v_accvgpr_read_b32 v136, a249
	v_accvgpr_read_b32 v143, a249
	v_accvgpr_read_b32 v142, a249
	v_accvgpr_read_b32 v141, a249
	v_accvgpr_read_b32 v140, a249
	v_accvgpr_read_b32 v147, a249
	v_accvgpr_read_b32 v146, a249
	v_accvgpr_read_b32 v145, a249
	v_accvgpr_read_b32 v144, a249
	v_accvgpr_read_b32 v151, a249
	v_accvgpr_read_b32 v150, a249
	v_accvgpr_read_b32 v149, a249
	v_accvgpr_read_b32 v148, a249
	v_accvgpr_read_b32 v155, a249
	v_accvgpr_read_b32 v154, a249
	v_accvgpr_read_b32 v153, a249
	v_accvgpr_read_b32 v152, a249
	v_accvgpr_read_b32 v159, a249
	v_accvgpr_read_b32 v158, a249
	v_accvgpr_read_b32 v157, a249
	v_accvgpr_read_b32 v156, a249
	v_accvgpr_read_b32 v163, a249
	v_accvgpr_read_b32 v162, a249
	v_accvgpr_read_b32 v161, a249
	v_accvgpr_read_b32 v160, a249
	v_accvgpr_read_b32 v167, a249
	v_accvgpr_read_b32 v166, a249
	v_accvgpr_read_b32 v165, a249
	v_accvgpr_read_b32 v164, a249
	v_accvgpr_read_b32 v171, a249
	v_accvgpr_read_b32 v170, a249
	v_accvgpr_read_b32 v169, a249
	v_accvgpr_read_b32 v168, a249
	v_accvgpr_read_b32 v175, a249
	v_accvgpr_read_b32 v174, a249
	v_accvgpr_read_b32 v173, a249
	v_accvgpr_read_b32 v172, a249
	v_accvgpr_read_b32 v179, a249
	v_accvgpr_read_b32 v178, a249
	v_accvgpr_read_b32 v177, a249
	v_accvgpr_read_b32 v176, a249
	v_accvgpr_read_b32 v183, a249
	v_accvgpr_read_b32 v182, a249
	v_accvgpr_read_b32 v181, a249
	v_accvgpr_read_b32 v180, a249
	v_accvgpr_read_b32 v187, a249
	v_accvgpr_read_b32 v186, a249
	v_accvgpr_read_b32 v185, a249
	v_accvgpr_read_b32 v184, a249
	v_accvgpr_read_b32 v195, a249
	v_accvgpr_read_b32 v194, a249
	v_accvgpr_read_b32 v193, a249
	v_accvgpr_read_b32 v192, a249
	v_accvgpr_read_b32 v191, a249
	v_accvgpr_read_b32 v190, a249
	v_accvgpr_read_b32 v189, a249
	v_accvgpr_read_b32 v188, a249
	v_accvgpr_read_b32 v199, a249
	v_accvgpr_read_b32 v198, a249
	v_accvgpr_read_b32 v197, a249
	v_accvgpr_read_b32 v196, a249
	v_accvgpr_read_b32 v203, a249
	v_accvgpr_read_b32 v202, a249
	v_accvgpr_read_b32 v201, a249
	v_accvgpr_read_b32 v200, a249
	v_accvgpr_read_b32 v207, a249
	v_accvgpr_read_b32 v206, a249
	v_accvgpr_read_b32 v205, a249
	v_accvgpr_read_b32 v204, a249
	v_accvgpr_read_b32 v211, a249
	v_accvgpr_read_b32 v210, a249
	v_accvgpr_read_b32 v209, a249
	v_accvgpr_read_b32 v208, a249
	v_accvgpr_read_b32 v215, a249
	v_accvgpr_read_b32 v214, a249
	v_accvgpr_read_b32 v213, a249
	v_accvgpr_read_b32 v212, a249
	v_accvgpr_read_b32 v219, a249
	v_accvgpr_read_b32 v218, a249
	v_accvgpr_read_b32 v217, a249
	v_accvgpr_read_b32 v216, a249
	v_accvgpr_read_b32 v223, a249
	v_accvgpr_read_b32 v222, a249
	v_accvgpr_read_b32 v221, a249
	v_accvgpr_read_b32 v220, a249
	v_accvgpr_read_b32 v227, a249
	v_accvgpr_read_b32 v226, a249
	v_accvgpr_read_b32 v225, a249
	v_accvgpr_read_b32 v224, a249
	v_accvgpr_read_b32 v231, a249
	v_accvgpr_read_b32 v230, a249
	v_accvgpr_read_b32 v229, a249
	v_accvgpr_read_b32 v228, a249
	v_accvgpr_read_b32 v235, a249
	v_accvgpr_read_b32 v234, a249
	v_accvgpr_read_b32 v233, a249
	v_accvgpr_read_b32 v232, a249
	v_accvgpr_read_b32 v239, a249
	v_accvgpr_read_b32 v238, a249
	v_accvgpr_read_b32 v237, a249
	v_accvgpr_read_b32 v236, a249
	v_accvgpr_read_b32 v243, a249
	v_accvgpr_read_b32 v242, a249
	v_accvgpr_read_b32 v241, a249
	v_accvgpr_read_b32 v240, a249
	v_accvgpr_read_b32 v247, a249
	v_accvgpr_read_b32 v246, a249
	v_accvgpr_read_b32 v245, a249
	v_accvgpr_read_b32 v244, a249
	v_accvgpr_read_b32 v7, a249
	v_accvgpr_read_b32 v6, a249
	v_accvgpr_read_b32 v5, a249
	v_accvgpr_read_b32 v4, a249
.LBB0_7:                                ; %._crit_edge.loopexit.peel.begin
	s_and_b32 s20, s15, 1
	s_waitcnt lgkmcnt(14)
	v_mfma_f32_16x16x32_f16 v[0:3], a[88:91], a[120:123], v[0:3]
	v_mfma_f32_16x16x32_f16 v[0:3], a[92:95], a[124:127], v[0:3]
	s_nop 7
	v_accvgpr_write_b32 a195, v3
	v_accvgpr_write_b32 a194, v2
	v_accvgpr_write_b32 a193, v1
	v_accvgpr_write_b32 a192, v0
	s_waitcnt lgkmcnt(13)
	v_mfma_f32_16x16x32_f16 v[4:7], a[80:83], a[120:123], v[4:7]
	s_waitcnt lgkmcnt(12)
	v_mfma_f32_16x16x32_f16 v[4:7], a[84:87], a[124:127], v[4:7]
	s_waitcnt lgkmcnt(11)
	v_mfma_f32_16x16x32_f16 v[8:11], a[64:67], a[120:123], v[244:247]
	s_waitcnt lgkmcnt(10)
	v_mfma_f32_16x16x32_f16 v[244:247], a[68:71], a[124:127], v[8:11]
	s_waitcnt lgkmcnt(9)
	v_mfma_f32_16x16x32_f16 v[8:11], a[56:59], a[120:123], v[240:243]
	s_waitcnt lgkmcnt(8)
	v_mfma_f32_16x16x32_f16 v[240:243], a[60:63], a[124:127], v[8:11]
	s_waitcnt lgkmcnt(7)
	v_mfma_f32_16x16x32_f16 v[8:11], a[48:51], a[120:123], v[236:239]
	s_waitcnt lgkmcnt(6)
	v_mfma_f32_16x16x32_f16 v[236:239], a[52:55], a[124:127], v[8:11]
	s_waitcnt lgkmcnt(5)
	v_mfma_f32_16x16x32_f16 v[8:11], a[40:43], a[120:123], v[232:235]
	s_waitcnt lgkmcnt(4)
	v_mfma_f32_16x16x32_f16 v[232:235], a[44:47], a[124:127], v[8:11]
	s_waitcnt lgkmcnt(3)
	v_mfma_f32_16x16x32_f16 v[8:11], a[32:35], a[120:123], v[228:231]
	s_waitcnt lgkmcnt(2)
	v_mfma_f32_16x16x32_f16 v[228:231], a[36:39], a[124:127], v[8:11]
	s_waitcnt lgkmcnt(1)
	v_mfma_f32_16x16x32_f16 v[8:11], a[24:27], a[120:123], v[224:227]
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_16x16x32_f16 v[224:227], a[28:31], a[124:127], v[8:11]
	v_mfma_f32_16x16x32_f16 v[8:11], a[88:91], a[112:115], v[220:223]
	v_mfma_f32_16x16x32_f16 v[220:223], a[92:95], a[116:119], v[8:11]
	v_mfma_f32_16x16x32_f16 v[8:11], a[80:83], a[112:115], v[216:219]
	v_mfma_f32_16x16x32_f16 v[216:219], a[84:87], a[116:119], v[8:11]
	v_mfma_f32_16x16x32_f16 v[8:11], a[64:67], a[112:115], v[212:215]
	v_mfma_f32_16x16x32_f16 v[212:215], a[68:71], a[116:119], v[8:11]
	v_mfma_f32_16x16x32_f16 v[8:11], a[56:59], a[112:115], v[208:211]
	v_mfma_f32_16x16x32_f16 v[208:211], a[60:63], a[116:119], v[8:11]
	v_mfma_f32_16x16x32_f16 v[8:11], a[48:51], a[112:115], v[204:207]
	v_mfma_f32_16x16x32_f16 v[204:207], a[52:55], a[116:119], v[8:11]
	v_mfma_f32_16x16x32_f16 v[8:11], a[40:43], a[112:115], v[200:203]
	v_mfma_f32_16x16x32_f16 v[200:203], a[44:47], a[116:119], v[8:11]
	v_mfma_f32_16x16x32_f16 v[8:11], a[32:35], a[112:115], v[196:199]
	v_mfma_f32_16x16x32_f16 v[196:199], a[36:39], a[116:119], v[8:11]
	v_mfma_f32_16x16x32_f16 v[8:11], a[24:27], a[112:115], v[188:191]
	v_mfma_f32_16x16x32_f16 v[188:191], a[28:31], a[116:119], v[8:11]
	v_mfma_f32_16x16x32_f16 v[8:11], a[88:91], a[104:107], v[192:195]
	v_mfma_f32_16x16x32_f16 v[192:195], a[92:95], a[108:111], v[8:11]
	v_mfma_f32_16x16x32_f16 v[8:11], a[80:83], a[104:107], v[184:187]
	v_mfma_f32_16x16x32_f16 v[184:187], a[84:87], a[108:111], v[8:11]
	v_mfma_f32_16x16x32_f16 v[8:11], a[64:67], a[104:107], v[180:183]
	v_mfma_f32_16x16x32_f16 v[180:183], a[68:71], a[108:111], v[8:11]
	v_mfma_f32_16x16x32_f16 v[8:11], a[56:59], a[104:107], v[176:179]
	v_mfma_f32_16x16x32_f16 v[176:179], a[60:63], a[108:111], v[8:11]
	v_mfma_f32_16x16x32_f16 v[8:11], a[48:51], a[104:107], v[172:175]
	v_mfma_f32_16x16x32_f16 v[172:175], a[52:55], a[108:111], v[8:11]
	v_mfma_f32_16x16x32_f16 v[8:11], a[40:43], a[104:107], v[168:171]
	v_mfma_f32_16x16x32_f16 v[168:171], a[44:47], a[108:111], v[8:11]
	v_mfma_f32_16x16x32_f16 v[8:11], a[32:35], a[104:107], v[164:167]
	v_mfma_f32_16x16x32_f16 v[164:167], a[36:39], a[108:111], v[8:11]
	v_mfma_f32_16x16x32_f16 v[8:11], a[24:27], a[104:107], v[160:163]
	v_mfma_f32_16x16x32_f16 v[160:163], a[28:31], a[108:111], v[8:11]
	v_mfma_f32_16x16x32_f16 v[8:11], a[88:91], a[96:99], v[156:159]
	v_mfma_f32_16x16x32_f16 v[156:159], a[92:95], a[100:103], v[8:11]
	v_mfma_f32_16x16x32_f16 v[8:11], a[80:83], a[96:99], v[152:155]
	v_mfma_f32_16x16x32_f16 v[152:155], a[84:87], a[100:103], v[8:11]
	v_mfma_f32_16x16x32_f16 v[8:11], a[64:67], a[96:99], v[148:151]
	v_mfma_f32_16x16x32_f16 v[148:151], a[68:71], a[100:103], v[8:11]
	v_mfma_f32_16x16x32_f16 v[8:11], a[56:59], a[96:99], v[144:147]
	v_mfma_f32_16x16x32_f16 v[144:147], a[60:63], a[100:103], v[8:11]
	v_mfma_f32_16x16x32_f16 v[8:11], a[48:51], a[96:99], v[140:143]
	v_mfma_f32_16x16x32_f16 v[140:143], a[52:55], a[100:103], v[8:11]
	v_mfma_f32_16x16x32_f16 v[8:11], a[40:43], a[96:99], v[136:139]
	v_mfma_f32_16x16x32_f16 v[136:139], a[44:47], a[100:103], v[8:11]
	v_mfma_f32_16x16x32_f16 v[8:11], a[32:35], a[96:99], v[132:135]
	v_mfma_f32_16x16x32_f16 v[132:135], a[36:39], a[100:103], v[8:11]
	v_mfma_f32_16x16x32_f16 v[8:11], a[24:27], a[96:99], v[128:131]
	v_mfma_f32_16x16x32_f16 v[128:131], a[28:31], a[100:103], v[8:11]
	v_mfma_f32_16x16x32_f16 v[8:11], a[88:91], a[16:19], v[124:127]
	v_mfma_f32_16x16x32_f16 v[124:127], a[92:95], a[20:23], v[8:11]
	v_mfma_f32_16x16x32_f16 v[8:11], a[80:83], a[16:19], v[120:123]
	v_mfma_f32_16x16x32_f16 v[120:123], a[84:87], a[20:23], v[8:11]
	v_mfma_f32_16x16x32_f16 v[8:11], a[64:67], a[16:19], v[116:119]
	v_mfma_f32_16x16x32_f16 v[116:119], a[68:71], a[20:23], v[8:11]
	v_mfma_f32_16x16x32_f16 v[8:11], a[56:59], a[16:19], v[112:115]
	v_mfma_f32_16x16x32_f16 v[112:115], a[60:63], a[20:23], v[8:11]
	v_mfma_f32_16x16x32_f16 v[8:11], a[48:51], a[16:19], v[108:111]
	v_mfma_f32_16x16x32_f16 v[108:111], a[52:55], a[20:23], v[8:11]
	v_mfma_f32_16x16x32_f16 v[8:11], a[40:43], a[16:19], v[104:107]
	v_mfma_f32_16x16x32_f16 v[104:107], a[44:47], a[20:23], v[8:11]
	v_mfma_f32_16x16x32_f16 v[8:11], a[32:35], a[16:19], v[100:103]
	v_mfma_f32_16x16x32_f16 v[100:103], a[36:39], a[20:23], v[8:11]
	v_mfma_f32_16x16x32_f16 v[8:11], a[24:27], a[16:19], v[96:99]
	v_mfma_f32_16x16x32_f16 v[96:99], a[28:31], a[20:23], v[8:11]
	v_mfma_f32_16x16x32_f16 v[8:11], a[88:91], a[0:3], v[92:95]
	v_mfma_f32_16x16x32_f16 v[92:95], a[92:95], a[4:7], v[8:11]
	v_mfma_f32_16x16x32_f16 v[8:11], a[80:83], a[0:3], v[88:91]
	v_mfma_f32_16x16x32_f16 v[88:91], a[84:87], a[4:7], v[8:11]
	v_mfma_f32_16x16x32_f16 v[8:11], a[64:67], a[0:3], v[84:87]
	v_mfma_f32_16x16x32_f16 v[84:87], a[68:71], a[4:7], v[8:11]
	v_mfma_f32_16x16x32_f16 v[8:11], a[56:59], a[0:3], v[80:83]
	v_mfma_f32_16x16x32_f16 v[80:83], a[60:63], a[4:7], v[8:11]
	v_mfma_f32_16x16x32_f16 v[8:11], a[48:51], a[0:3], v[76:79]
	v_mfma_f32_16x16x32_f16 v[76:79], a[52:55], a[4:7], v[8:11]
	v_mfma_f32_16x16x32_f16 v[8:11], a[40:43], a[0:3], v[72:75]
	v_mfma_f32_16x16x32_f16 v[72:75], a[44:47], a[4:7], v[8:11]
	v_mfma_f32_16x16x32_f16 v[8:11], a[32:35], a[0:3], v[68:71]
	v_mfma_f32_16x16x32_f16 v[68:71], a[36:39], a[4:7], v[8:11]
	v_mfma_f32_16x16x32_f16 v[8:11], a[24:27], a[0:3], v[64:67]
	v_mfma_f32_16x16x32_f16 v[64:67], a[28:31], a[4:7], v[8:11]
	v_mfma_f32_16x16x32_f16 a[136:139], a[88:91], a[72:75], a[250:253]
	v_mfma_f32_16x16x32_f16 a[136:139], a[92:95], a[76:79], a[136:139]
	v_mfma_f32_16x16x32_f16 v[8:11], a[80:83], a[72:75], v[56:59]
	v_mfma_f32_16x16x32_f16 v[0:3], a[84:87], a[76:79], v[8:11]
	s_nop 7
	v_accvgpr_write_b32 a143, v3
	v_accvgpr_write_b32 a142, v2
	v_accvgpr_write_b32 a141, v1
	v_accvgpr_write_b32 a140, v0
	v_mfma_f32_16x16x32_f16 v[8:11], a[64:67], a[72:75], v[52:55]
	v_mfma_f32_16x16x32_f16 v[0:3], a[68:71], a[76:79], v[8:11]
	s_nop 7
	v_accvgpr_write_b32 a147, v3
	v_accvgpr_write_b32 a146, v2
	v_accvgpr_write_b32 a145, v1
	v_accvgpr_write_b32 a144, v0
	v_mfma_f32_16x16x32_f16 v[8:11], a[56:59], a[72:75], v[48:51]
	v_mfma_f32_16x16x32_f16 v[0:3], a[60:63], a[76:79], v[8:11]
	s_nop 7
	v_accvgpr_write_b32 a151, v3
	v_accvgpr_write_b32 a150, v2
	v_accvgpr_write_b32 a149, v1
	v_accvgpr_write_b32 a148, v0
	v_mfma_f32_16x16x32_f16 v[8:11], a[48:51], a[72:75], v[44:47]
	v_mfma_f32_16x16x32_f16 v[0:3], a[52:55], a[76:79], v[8:11]
	s_nop 7
	v_accvgpr_write_b32 a155, v3
	v_accvgpr_write_b32 a154, v2
	v_accvgpr_write_b32 a153, v1
	v_accvgpr_write_b32 a152, v0
	v_mfma_f32_16x16x32_f16 v[8:11], a[40:43], a[72:75], v[40:43]
	v_mfma_f32_16x16x32_f16 v[0:3], a[44:47], a[76:79], v[8:11]
	s_nop 7
	v_accvgpr_write_b32 a159, v3
	v_accvgpr_write_b32 a158, v2
	v_accvgpr_write_b32 a157, v1
	v_accvgpr_write_b32 a156, v0
	v_mfma_f32_16x16x32_f16 v[8:11], a[32:35], a[72:75], v[36:39]
	v_mfma_f32_16x16x32_f16 v[0:3], a[36:39], a[76:79], v[8:11]
	s_nop 7
	v_accvgpr_write_b32 a163, v3
	v_accvgpr_write_b32 a162, v2
	v_accvgpr_write_b32 a161, v1
	v_accvgpr_write_b32 a160, v0
	v_mfma_f32_16x16x32_f16 v[8:11], a[24:27], a[72:75], v[32:35]
	v_mfma_f32_16x16x32_f16 v[0:3], a[28:31], a[76:79], v[8:11]
	s_nop 7
	v_accvgpr_write_b32 a167, v3
	v_accvgpr_write_b32 a166, v2
	v_accvgpr_write_b32 a165, v1
	v_accvgpr_write_b32 a164, v0
	v_mfma_f32_16x16x32_f16 v[8:11], a[88:91], a[8:11], v[28:31]
	v_mfma_f32_16x16x32_f16 v[0:3], a[92:95], a[12:15], v[8:11]
	s_nop 7
	v_accvgpr_write_b32 a171, v3
	v_accvgpr_write_b32 a170, v2
	v_accvgpr_write_b32 a169, v1
	v_accvgpr_write_b32 a168, v0
	v_mfma_f32_16x16x32_f16 v[8:11], a[80:83], a[8:11], v[24:27]
	v_mfma_f32_16x16x32_f16 v[0:3], a[84:87], a[12:15], v[8:11]
	s_nop 7
	v_accvgpr_write_b32 a175, v3
	v_accvgpr_write_b32 a174, v2
	v_accvgpr_write_b32 a173, v1
	v_accvgpr_write_b32 a172, v0
	v_mfma_f32_16x16x32_f16 v[8:11], a[64:67], a[8:11], v[20:23]
	v_mfma_f32_16x16x32_f16 v[0:3], a[68:71], a[12:15], v[8:11]
	s_nop 7
	v_accvgpr_write_b32 a179, v3
	v_accvgpr_write_b32 a178, v2
	v_accvgpr_write_b32 a177, v1
	v_accvgpr_write_b32 a176, v0
	v_mfma_f32_16x16x32_f16 v[8:11], a[56:59], a[8:11], v[16:19]
	v_mfma_f32_16x16x32_f16 v[0:3], a[60:63], a[12:15], v[8:11]
	s_nop 7
	v_accvgpr_write_b32 a183, v3
	v_accvgpr_write_b32 a182, v2
	v_accvgpr_write_b32 a181, v1
	v_accvgpr_write_b32 a180, v0
	v_mfma_f32_16x16x32_f16 v[8:11], a[48:51], a[8:11], v[12:15]
	v_mfma_f32_16x16x32_f16 v[0:3], a[52:55], a[12:15], v[8:11]
	s_nop 7
	v_accvgpr_write_b32 a187, v3
	v_accvgpr_write_b32 a186, v2
	v_accvgpr_write_b32 a185, v1
	v_accvgpr_write_b32 a184, v0
	v_accvgpr_mov_b32 a0, a132
	v_accvgpr_mov_b32 a1, a133
	v_accvgpr_mov_b32 a2, a134
	v_accvgpr_mov_b32 a3, a135
	s_nop 1
	v_mfma_f32_16x16x32_f16 a[0:3], a[40:43], a[8:11], a[0:3]
	v_mfma_f32_16x16x32_f16 a[132:135], a[44:47], a[12:15], a[0:3]
	s_nop 6
	v_accvgpr_mov_b32 a0, a128
	v_accvgpr_mov_b32 a1, a129
	v_accvgpr_mov_b32 a2, a130
	v_accvgpr_mov_b32 a3, a131
	s_nop 1
	v_mfma_f32_16x16x32_f16 a[0:3], a[32:35], a[8:11], a[0:3]
	v_mfma_f32_16x16x32_f16 a[188:191], a[36:39], a[12:15], a[0:3]
	v_mfma_f32_16x16x32_f16 a[128:131], a[24:27], a[8:11], a[246:249]
	v_mfma_f32_16x16x32_f16 a[128:131], a[28:31], a[12:15], a[128:131]
	; wait_asyncmark(0)
	s_waitcnt vmcnt(0)
	s_barrier
	s_cmp_eq_u32 s15, s7
	s_cselect_b64 vcc, -1, 0
	s_mul_i32 s7, s20, 0x8400
	s_add_i32 s7, s7, 0
	s_and_b32 s17, s17, 0xffff
	s_mov_b32 s19, 0x27000
	s_mov_b32 s18, 0x7ffffffe
	s_add_i32 s6, s7, s6
	v_bfrev_b32_e32 v0, 1
	v_cndmask_b32_e32 v1, v252, v0, vcc
	s_mov_b32 m0, s6
	s_nop 0
	buffer_load_dwordx4 v1, s[16:19], 0 offen lds
	s_add_i32 m0, s6, 0x1080
	v_accvgpr_read_b32 v1, a211
	v_cndmask_b32_e32 v1, v1, v0, vcc
	buffer_load_dwordx4 v1, s[16:19], 0 offen lds
	s_add_i32 m0, s6, 0x2100
	v_accvgpr_read_b32 v1, a212
	v_cndmask_b32_e32 v1, v1, v0, vcc
	buffer_load_dwordx4 v1, s[16:19], 0 offen lds
	s_add_i32 m0, s6, 0x3180
	v_accvgpr_read_b32 v1, a213
	v_cndmask_b32_e32 v1, v1, v0, vcc
	buffer_load_dwordx4 v1, s[16:19], 0 offen lds
	s_add_i32 m0, s6, 0x4200
	v_cndmask_b32_e32 v1, v253, v0, vcc
	buffer_load_dwordx4 v1, s[16:19], 0 offen lds
	s_add_i32 m0, s6, 0x5280
	v_cndmask_b32_e32 v1, v254, v0, vcc
	buffer_load_dwordx4 v1, s[16:19], 0 offen lds
	s_add_i32 m0, s6, 0x6300
	v_accvgpr_read_b32 v1, a214
	v_cndmask_b32_e32 v1, v1, v0, vcc
	buffer_load_dwordx4 v1, s[16:19], 0 offen lds
	s_add_i32 m0, s6, 0x7380
	v_accvgpr_read_b32 v1, a215
	v_cndmask_b32_e32 v1, v1, v0, vcc
	buffer_load_dwordx4 v1, s[16:19], 0 offen lds
	s_and_b32 s9, s9, 0xffff
	s_mov_b32 s10, s18
	s_mov_b32 s11, s19
	s_add_i32 m0, s6, 0x107e0
	v_accvgpr_read_b32 v1, a216
	v_cndmask_b32_e32 v1, v1, v0, vcc
	buffer_load_dwordx4 v1, s[8:11], 0 offen lds
	s_add_i32 m0, s6, 0x11860
	v_accvgpr_read_b32 v1, a217
	v_cndmask_b32_e32 v1, v1, v0, vcc
	buffer_load_dwordx4 v1, s[8:11], 0 offen lds
	s_add_i32 m0, s6, 0x128e0
	v_accvgpr_read_b32 v1, a218
	v_cndmask_b32_e32 v1, v1, v0, vcc
	buffer_load_dwordx4 v1, s[8:11], 0 offen lds
	s_add_i32 m0, s6, 0x13960
	v_accvgpr_read_b32 v1, a219
	v_cndmask_b32_e32 v1, v1, v0, vcc
	buffer_load_dwordx4 v1, s[8:11], 0 offen lds
	s_add_i32 m0, s6, 0x149e0
	v_accvgpr_read_b32 v1, a220
	v_cndmask_b32_e32 v1, v1, v0, vcc
	buffer_load_dwordx4 v1, s[8:11], 0 offen lds
	s_add_i32 m0, s6, 0x15a60
	v_accvgpr_read_b32 v1, a221
	v_cndmask_b32_e32 v1, v1, v0, vcc
	buffer_load_dwordx4 v1, s[8:11], 0 offen lds
	s_add_i32 m0, s6, 0x16ae0
	v_accvgpr_read_b32 v1, a222
	v_cndmask_b32_e32 v1, v1, v0, vcc
	buffer_load_dwordx4 v1, s[8:11], 0 offen lds
	s_add_i32 m0, s6, 0x17b60
	v_accvgpr_read_b32 v1, a223
	v_cndmask_b32_e32 v0, v1, v0, vcc
	buffer_load_dwordx4 v0, s[8:11], 0 offen lds
	; asyncmark
	s_lshl_b32 s6, s20, 14
	s_xor_b32 s6, s6, 0x4000
	s_lshr_b32 s7, s6, 5
	s_or_b32 s7, s7, s6
	s_lshl1_add_u32 s6, s7, 0
	s_barrier
	v_accvgpr_read_b32 v0, a224
	v_add_u32_e32 v0, s6, v0
	ds_read_b128 a[120:123], v0
	ds_read_b128 a[124:127], v0 offset:64
	ds_read_b128 a[112:115], v0 offset:256
	ds_read_b128 a[116:119], v0 offset:320
	ds_read_b128 a[104:107], v0 offset:512
	ds_read_b128 a[108:111], v0 offset:576
	ds_read_b128 a[96:99], v0 offset:768
	ds_read_b128 a[100:103], v0 offset:832
	ds_read_b128 a[16:19], v0 offset:16896
	ds_read_b128 a[20:23], v0 offset:16960
	ds_read_b128 a[0:3], v0 offset:17152
	ds_read_b128 a[4:7], v0 offset:17216
	ds_read_b128 a[72:75], v0 offset:17408
	ds_read_b128 a[76:79], v0 offset:17472
	ds_read_b128 a[8:11], v0 offset:17664
	ds_read_b128 a[12:15], v0 offset:17728
	v_accvgpr_read_b32 v0, a225
	v_add_u32_e32 v0, s6, v0
	v_add_u32_e32 v0, 0x107e0, v0
	ds_read_b128 a[88:91], v0
	ds_read_b128 a[92:95], v0 offset:64
	ds_read_b128 a[80:83], v0 offset:256
	ds_read_b128 a[84:87], v0 offset:320
	ds_read_b128 a[64:67], v0 offset:512
	ds_read_b128 a[68:71], v0 offset:576
	ds_read_b128 a[56:59], v0 offset:768
	ds_read_b128 a[60:63], v0 offset:832
	ds_read_b128 a[48:51], v0 offset:16896
	ds_read_b128 a[52:55], v0 offset:16960
	ds_read_b128 a[40:43], v0 offset:17152
	ds_read_b128 a[44:47], v0 offset:17216
	ds_read_b128 a[32:35], v0 offset:17408
	ds_read_b128 a[36:39], v0 offset:17472
	ds_read_b128 a[24:27], v0 offset:17664
	ds_read_b128 a[28:31], v0 offset:17728
.LBB0_8:                                ; %Flow388
	s_load_dword s4, s[4:5], 0x2c
	s_lshr_b32 s3, s3, 6
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_16x16x32_f16 a[192:195], a[88:91], a[120:123], a[192:195]
	v_mfma_f32_16x16x32_f16 a[192:195], a[92:95], a[124:127], a[192:195]
	v_mfma_f32_16x16x32_f16 v[4:7], a[80:83], a[120:123], v[4:7]
	v_mfma_f32_16x16x32_f16 v[0:3], a[84:87], a[124:127], v[4:7]
	s_nop 7
	v_accvgpr_write_b32 a199, v3
	v_accvgpr_write_b32 a198, v2
	v_accvgpr_write_b32 a197, v1
	v_accvgpr_write_b32 a196, v0
	v_mfma_f32_16x16x32_f16 v[244:247], a[64:67], a[120:123], v[244:247]
	v_mfma_f32_16x16x32_f16 v[244:247], a[68:71], a[124:127], v[244:247]
	v_mfma_f32_16x16x32_f16 v[240:243], a[56:59], a[120:123], v[240:243]
	v_mfma_f32_16x16x32_f16 v[240:243], a[60:63], a[124:127], v[240:243]
	v_mfma_f32_16x16x32_f16 v[236:239], a[48:51], a[120:123], v[236:239]
	v_mfma_f32_16x16x32_f16 v[236:239], a[52:55], a[124:127], v[236:239]
	v_mfma_f32_16x16x32_f16 v[232:235], a[40:43], a[120:123], v[232:235]
	v_mfma_f32_16x16x32_f16 v[232:235], a[44:47], a[124:127], v[232:235]
	v_mfma_f32_16x16x32_f16 v[228:231], a[32:35], a[120:123], v[228:231]
	v_mfma_f32_16x16x32_f16 v[228:231], a[36:39], a[124:127], v[228:231]
	v_mfma_f32_16x16x32_f16 v[224:227], a[24:27], a[120:123], v[224:227]
	v_mfma_f32_16x16x32_f16 v[224:227], a[28:31], a[124:127], v[224:227]
	v_mfma_f32_16x16x32_f16 v[220:223], a[88:91], a[112:115], v[220:223]
	v_mfma_f32_16x16x32_f16 v[220:223], a[92:95], a[116:119], v[220:223]
	v_mfma_f32_16x16x32_f16 v[216:219], a[80:83], a[112:115], v[216:219]
	v_mfma_f32_16x16x32_f16 v[216:219], a[84:87], a[116:119], v[216:219]
	v_mfma_f32_16x16x32_f16 v[212:215], a[64:67], a[112:115], v[212:215]
	v_mfma_f32_16x16x32_f16 v[212:215], a[68:71], a[116:119], v[212:215]
	v_mfma_f32_16x16x32_f16 v[208:211], a[56:59], a[112:115], v[208:211]
	v_mfma_f32_16x16x32_f16 v[208:211], a[60:63], a[116:119], v[208:211]
	v_mfma_f32_16x16x32_f16 v[204:207], a[48:51], a[112:115], v[204:207]
	v_mfma_f32_16x16x32_f16 v[204:207], a[52:55], a[116:119], v[204:207]
	v_mfma_f32_16x16x32_f16 v[200:203], a[40:43], a[112:115], v[200:203]
	v_mfma_f32_16x16x32_f16 v[200:203], a[44:47], a[116:119], v[200:203]
	v_mfma_f32_16x16x32_f16 v[196:199], a[32:35], a[112:115], v[196:199]
	v_mfma_f32_16x16x32_f16 v[196:199], a[36:39], a[116:119], v[196:199]
	v_mfma_f32_16x16x32_f16 v[188:191], a[24:27], a[112:115], v[188:191]
	v_mfma_f32_16x16x32_f16 v[188:191], a[28:31], a[116:119], v[188:191]
	v_mfma_f32_16x16x32_f16 v[192:195], a[88:91], a[104:107], v[192:195]
	v_mfma_f32_16x16x32_f16 v[0:3], a[92:95], a[108:111], v[192:195]
	s_nop 7
	v_accvgpr_write_b32 a115, v3
	v_accvgpr_write_b32 a114, v2
	v_accvgpr_write_b32 a113, v1
	v_accvgpr_write_b32 a112, v0
	v_mfma_f32_16x16x32_f16 v[184:187], a[80:83], a[104:107], v[184:187]
	v_mfma_f32_16x16x32_f16 v[184:187], a[84:87], a[108:111], v[184:187]
	v_mfma_f32_16x16x32_f16 v[180:183], a[64:67], a[104:107], v[180:183]
	v_mfma_f32_16x16x32_f16 v[180:183], a[68:71], a[108:111], v[180:183]
	v_mfma_f32_16x16x32_f16 v[176:179], a[56:59], a[104:107], v[176:179]
	v_mfma_f32_16x16x32_f16 v[176:179], a[60:63], a[108:111], v[176:179]
	v_mfma_f32_16x16x32_f16 v[172:175], a[48:51], a[104:107], v[172:175]
	v_mfma_f32_16x16x32_f16 v[172:175], a[52:55], a[108:111], v[172:175]
	v_mfma_f32_16x16x32_f16 v[168:171], a[40:43], a[104:107], v[168:171]
	v_mfma_f32_16x16x32_f16 v[168:171], a[44:47], a[108:111], v[168:171]
	v_mfma_f32_16x16x32_f16 v[164:167], a[32:35], a[104:107], v[164:167]
	v_mfma_f32_16x16x32_f16 v[164:167], a[36:39], a[108:111], v[164:167]
	v_mfma_f32_16x16x32_f16 v[160:163], a[24:27], a[104:107], v[160:163]
	v_mfma_f32_16x16x32_f16 v[160:163], a[28:31], a[108:111], v[160:163]
	v_mfma_f32_16x16x32_f16 v[156:159], a[88:91], a[96:99], v[156:159]
	v_mfma_f32_16x16x32_f16 v[192:195], a[92:95], a[100:103], v[156:159]
	v_mfma_f32_16x16x32_f16 v[152:155], a[80:83], a[96:99], v[152:155]
	v_mfma_f32_16x16x32_f16 v[156:159], a[84:87], a[100:103], v[152:155]
	v_mfma_f32_16x16x32_f16 v[148:151], a[64:67], a[96:99], v[148:151]
	v_mfma_f32_16x16x32_f16 v[152:155], a[68:71], a[100:103], v[148:151]
	v_mfma_f32_16x16x32_f16 v[144:147], a[56:59], a[96:99], v[144:147]
	v_mfma_f32_16x16x32_f16 v[148:151], a[60:63], a[100:103], v[144:147]
	v_mfma_f32_16x16x32_f16 v[140:143], a[48:51], a[96:99], v[140:143]
	v_mfma_f32_16x16x32_f16 v[144:147], a[52:55], a[100:103], v[140:143]
	v_mfma_f32_16x16x32_f16 v[136:139], a[40:43], a[96:99], v[136:139]
	v_mfma_f32_16x16x32_f16 v[140:143], a[44:47], a[100:103], v[136:139]
	v_mfma_f32_16x16x32_f16 v[132:135], a[32:35], a[96:99], v[132:135]
	v_mfma_f32_16x16x32_f16 v[136:139], a[36:39], a[100:103], v[132:135]
	v_mfma_f32_16x16x32_f16 v[128:131], a[24:27], a[96:99], v[128:131]
	v_mfma_f32_16x16x32_f16 v[132:135], a[28:31], a[100:103], v[128:131]
	v_mfma_f32_16x16x32_f16 v[124:127], a[88:91], a[16:19], v[124:127]
	v_mfma_f32_16x16x32_f16 v[248:251], a[92:95], a[20:23], v[124:127]
	v_mfma_f32_16x16x32_f16 v[120:123], a[80:83], a[16:19], v[120:123]
	v_mfma_f32_16x16x32_f16 v[128:131], a[84:87], a[20:23], v[120:123]
	v_mfma_f32_16x16x32_f16 v[116:119], a[64:67], a[16:19], v[116:119]
	v_mfma_f32_16x16x32_f16 v[44:47], a[68:71], a[20:23], v[116:119]
	v_mfma_f32_16x16x32_f16 v[112:115], a[56:59], a[16:19], v[112:115]
	v_mfma_f32_16x16x32_f16 v[16:19], a[60:63], a[20:23], v[112:115]
	v_mfma_f32_16x16x32_f16 v[108:111], a[48:51], a[16:19], v[108:111]
	v_mfma_f32_16x16x32_f16 v[20:23], a[52:55], a[20:23], v[108:111]
	v_mfma_f32_16x16x32_f16 v[104:107], a[40:43], a[16:19], v[104:107]
	v_mfma_f32_16x16x32_f16 v[24:27], a[44:47], a[20:23], v[104:107]
	v_mfma_f32_16x16x32_f16 v[100:103], a[32:35], a[16:19], v[100:103]
	v_mfma_f32_16x16x32_f16 v[28:31], a[36:39], a[20:23], v[100:103]
	v_mfma_f32_16x16x32_f16 v[96:99], a[24:27], a[16:19], v[96:99]
	v_mfma_f32_16x16x32_f16 v[32:35], a[28:31], a[20:23], v[96:99]
	v_mfma_f32_16x16x32_f16 v[92:95], a[88:91], a[0:3], v[92:95]
	v_mfma_f32_16x16x32_f16 v[36:39], a[92:95], a[4:7], v[92:95]
	v_mfma_f32_16x16x32_f16 v[88:91], a[80:83], a[0:3], v[88:91]
	v_mfma_f32_16x16x32_f16 v[40:43], a[84:87], a[4:7], v[88:91]
	v_mfma_f32_16x16x32_f16 v[84:87], a[64:67], a[0:3], v[84:87]
	v_mfma_f32_16x16x32_f16 v[12:15], a[68:71], a[4:7], v[84:87]
	v_mfma_f32_16x16x32_f16 v[80:83], a[56:59], a[0:3], v[80:83]
	v_mfma_f32_16x16x32_f16 v[48:51], a[60:63], a[4:7], v[80:83]
	v_mfma_f32_16x16x32_f16 v[76:79], a[48:51], a[0:3], v[76:79]
	v_mfma_f32_16x16x32_f16 v[52:55], a[52:55], a[4:7], v[76:79]
	v_mfma_f32_16x16x32_f16 v[72:75], a[40:43], a[0:3], v[72:75]
	v_mfma_f32_16x16x32_f16 v[56:59], a[44:47], a[4:7], v[72:75]
	v_mfma_f32_16x16x32_f16 v[68:71], a[32:35], a[0:3], v[68:71]
	v_mfma_f32_16x16x32_f16 v[60:63], a[36:39], a[4:7], v[68:71]
	v_mfma_f32_16x16x32_f16 v[64:67], a[24:27], a[0:3], v[64:67]
	v_mfma_f32_16x16x32_f16 v[64:67], a[28:31], a[4:7], v[64:67]
	v_accvgpr_read_b32 v0, a136
	v_accvgpr_read_b32 v1, a137
	v_accvgpr_read_b32 v2, a138
	v_accvgpr_read_b32 v3, a139
	s_nop 1
	v_mfma_f32_16x16x32_f16 v[68:71], a[88:91], a[72:75], v[0:3]
	v_mfma_f32_16x16x32_f16 v[68:71], a[92:95], a[76:79], v[68:71]
	s_nop 1
	v_accvgpr_read_b32 v0, a140
	v_accvgpr_read_b32 v1, a141
	v_accvgpr_read_b32 v2, a142
	v_accvgpr_read_b32 v3, a143
	s_nop 1
	v_mfma_f32_16x16x32_f16 v[72:75], a[80:83], a[72:75], v[0:3]
	v_mfma_f32_16x16x32_f16 v[72:75], a[84:87], a[76:79], v[72:75]
	s_nop 1
	v_accvgpr_read_b32 v0, a144
	v_accvgpr_read_b32 v1, a145
	v_accvgpr_read_b32 v2, a146
	v_accvgpr_read_b32 v3, a147
	s_nop 1
	v_mfma_f32_16x16x32_f16 v[76:79], a[64:67], a[72:75], v[0:3]
	v_mfma_f32_16x16x32_f16 v[76:79], a[68:71], a[76:79], v[76:79]
	s_nop 1
	v_accvgpr_read_b32 v0, a148
	v_accvgpr_read_b32 v1, a149
	v_accvgpr_read_b32 v2, a150
	v_accvgpr_read_b32 v3, a151
	s_nop 1
	v_mfma_f32_16x16x32_f16 v[80:83], a[56:59], a[72:75], v[0:3]
	v_mfma_f32_16x16x32_f16 v[80:83], a[60:63], a[76:79], v[80:83]
	s_nop 1
	v_accvgpr_read_b32 v0, a152
	v_accvgpr_read_b32 v1, a153
	v_accvgpr_read_b32 v2, a154
	v_accvgpr_read_b32 v3, a155
	s_nop 1
	v_mfma_f32_16x16x32_f16 v[84:87], a[48:51], a[72:75], v[0:3]
	v_mfma_f32_16x16x32_f16 v[84:87], a[52:55], a[76:79], v[84:87]
	s_nop 1
	v_accvgpr_read_b32 v0, a156
	v_accvgpr_read_b32 v1, a157
	v_accvgpr_read_b32 v2, a158
	v_accvgpr_read_b32 v3, a159
	s_nop 1
	v_mfma_f32_16x16x32_f16 v[88:91], a[40:43], a[72:75], v[0:3]
	v_mfma_f32_16x16x32_f16 v[88:91], a[44:47], a[76:79], v[88:91]
	s_nop 1
	v_accvgpr_read_b32 v0, a160
	v_accvgpr_read_b32 v1, a161
	v_accvgpr_read_b32 v2, a162
	v_accvgpr_read_b32 v3, a163
	s_nop 1
	v_mfma_f32_16x16x32_f16 v[92:95], a[32:35], a[72:75], v[0:3]
	v_mfma_f32_16x16x32_f16 v[92:95], a[36:39], a[76:79], v[92:95]
	s_nop 1
	v_accvgpr_read_b32 v0, a164
	v_accvgpr_read_b32 v1, a165
	v_accvgpr_read_b32 v2, a166
	v_accvgpr_read_b32 v3, a167
	s_nop 1
	v_mfma_f32_16x16x32_f16 v[96:99], a[24:27], a[72:75], v[0:3]
	v_mfma_f32_16x16x32_f16 v[96:99], a[28:31], a[76:79], v[96:99]
	s_nop 1
	v_accvgpr_read_b32 v0, a168
	v_accvgpr_read_b32 v1, a169
	v_accvgpr_read_b32 v2, a170
	v_accvgpr_read_b32 v3, a171
	s_nop 1
	v_mfma_f32_16x16x32_f16 v[100:103], a[88:91], a[8:11], v[0:3]
	v_mfma_f32_16x16x32_f16 v[100:103], a[92:95], a[12:15], v[100:103]
	s_nop 1
	v_accvgpr_read_b32 v0, a172
	v_accvgpr_read_b32 v1, a173
	v_accvgpr_read_b32 v2, a174
	v_accvgpr_read_b32 v3, a175
	s_nop 1
	v_mfma_f32_16x16x32_f16 v[104:107], a[80:83], a[8:11], v[0:3]
	v_mfma_f32_16x16x32_f16 v[104:107], a[84:87], a[12:15], v[104:107]
	s_nop 1
	v_accvgpr_read_b32 v0, a176
	v_accvgpr_read_b32 v1, a177
	v_accvgpr_read_b32 v2, a178
	v_accvgpr_read_b32 v3, a179
	s_nop 1
	v_mfma_f32_16x16x32_f16 v[108:111], a[64:67], a[8:11], v[0:3]
	v_mfma_f32_16x16x32_f16 v[108:111], a[68:71], a[12:15], v[108:111]
	s_nop 1
	v_accvgpr_read_b32 v0, a180
	v_accvgpr_read_b32 v1, a181
	v_accvgpr_read_b32 v2, a182
	v_accvgpr_read_b32 v3, a183
	s_nop 1
	v_mfma_f32_16x16x32_f16 v[112:115], a[56:59], a[8:11], v[0:3]
	v_mfma_f32_16x16x32_f16 v[112:115], a[60:63], a[12:15], v[112:115]
	s_nop 1
	v_accvgpr_read_b32 v0, a184
	v_accvgpr_read_b32 v1, a185
	v_accvgpr_read_b32 v2, a186
	v_accvgpr_read_b32 v3, a187
	s_nop 1
	v_mfma_f32_16x16x32_f16 v[116:119], a[48:51], a[8:11], v[0:3]
	v_mfma_f32_16x16x32_f16 v[116:119], a[52:55], a[12:15], v[116:119]
	s_nop 1
	v_accvgpr_read_b32 v0, a132
	v_accvgpr_read_b32 v1, a133
	v_accvgpr_read_b32 v2, a134
	v_accvgpr_read_b32 v3, a135
	s_nop 1
	v_mfma_f32_16x16x32_f16 v[120:123], a[40:43], a[8:11], v[0:3]
	v_mfma_f32_16x16x32_f16 v[120:123], a[44:47], a[12:15], v[120:123]
	s_nop 1
	v_accvgpr_read_b32 v0, a188
	v_accvgpr_read_b32 v1, a189
	v_accvgpr_read_b32 v2, a190
	v_accvgpr_read_b32 v3, a191
	s_nop 1
	v_mfma_f32_16x16x32_f16 v[124:127], a[32:35], a[8:11], v[0:3]
	v_mfma_f32_16x16x32_f16 v[8:11], a[36:39], a[12:15], v[124:127]
	s_nop 1
	v_accvgpr_read_b32 v0, a128
	v_accvgpr_read_b32 v1, a129
	v_accvgpr_read_b32 v2, a130
	v_accvgpr_read_b32 v3, a131
	s_nop 1
	v_mfma_f32_16x16x32_f16 v[124:127], a[24:27], a[8:11], v[0:3]
	v_mfma_f32_16x16x32_f16 v[4:7], a[28:31], a[12:15], v[124:127]
	s_nop 1
	v_accvgpr_read_b32 v0, a192
	v_accvgpr_read_b32 v1, a193
	v_accvgpr_read_b32 v2, a194
	v_accvgpr_read_b32 v3, a195
	s_nop 0
	v_cvt_pk_f16_f32 v126, v0, v1
	v_cvt_pk_f16_f32 v127, v2, v3
	v_accvgpr_read_b32 v0, a196
	v_accvgpr_read_b32 v1, a197
	v_accvgpr_read_b32 v2, a198
	v_accvgpr_read_b32 v3, a199
	v_cvt_pk_f16_f32 v124, v0, v1
	v_cvt_pk_f16_f32 v125, v2, v3
	v_cvt_pk_f16_f32 v252, v244, v245
	v_cvt_pk_f16_f32 v253, v246, v247
	v_cvt_pk_f16_f32 v244, v240, v241
	v_cvt_pk_f16_f32 v245, v242, v243
	v_cvt_pk_f16_f32 v240, v236, v237
	v_cvt_pk_f16_f32 v241, v238, v239
	v_cvt_pk_f16_f32 v236, v232, v233
	v_cvt_pk_f16_f32 v237, v234, v235
	v_cvt_pk_f16_f32 v232, v228, v229
	v_cvt_pk_f16_f32 v233, v230, v231
	v_cvt_pk_f16_f32 v228, v224, v225
	v_cvt_pk_f16_f32 v229, v226, v227
	v_cvt_pk_f16_f32 v224, v220, v221
	v_cvt_pk_f16_f32 v225, v222, v223
	v_cvt_pk_f16_f32 v220, v216, v217
	v_cvt_pk_f16_f32 v221, v218, v219
	v_cvt_pk_f16_f32 v216, v212, v213
	v_cvt_pk_f16_f32 v217, v214, v215
	v_cvt_pk_f16_f32 v212, v208, v209
	v_cvt_pk_f16_f32 v213, v210, v211
	v_cvt_pk_f16_f32 v208, v204, v205
	v_cvt_pk_f16_f32 v209, v206, v207
	v_cvt_pk_f16_f32 v204, v200, v201
	v_cvt_pk_f16_f32 v205, v202, v203
	v_cvt_pk_f16_f32 v200, v196, v197
	v_cvt_pk_f16_f32 v201, v198, v199
	v_cvt_pk_f16_f32 v196, v188, v189
	v_cvt_pk_f16_f32 v197, v190, v191
	v_accvgpr_read_b32 v0, a112
	v_accvgpr_read_b32 v1, a113
	v_accvgpr_read_b32 v2, a114
	v_accvgpr_read_b32 v3, a115
	v_cvt_pk_f16_f32 v190, v0, v1
	v_cvt_pk_f16_f32 v191, v2, v3
	v_cvt_pk_f16_f32 v188, v184, v185
	v_cvt_pk_f16_f32 v189, v186, v187
	v_cvt_pk_f16_f32 v184, v180, v181
	v_cvt_pk_f16_f32 v185, v182, v183
	v_cvt_pk_f16_f32 v180, v176, v177
	v_cvt_pk_f16_f32 v181, v178, v179
	v_cvt_pk_f16_f32 v176, v172, v173
	v_cvt_pk_f16_f32 v177, v174, v175
	v_cvt_pk_f16_f32 v172, v168, v169
	v_cvt_pk_f16_f32 v173, v170, v171
	v_cvt_pk_f16_f32 v168, v164, v165
	v_cvt_pk_f16_f32 v169, v166, v167
	v_cvt_pk_f16_f32 v164, v160, v161
	v_cvt_pk_f16_f32 v165, v162, v163
	v_cvt_pk_f16_f32 v162, v192, v193
	v_cvt_pk_f16_f32 v163, v194, v195
	v_cvt_pk_f16_f32 v160, v156, v157
	v_cvt_pk_f16_f32 v161, v158, v159
	v_cvt_pk_f16_f32 v158, v152, v153
	v_cvt_pk_f16_f32 v159, v154, v155
	v_cvt_pk_f16_f32 v156, v148, v149
	v_cvt_pk_f16_f32 v157, v150, v151
	v_cvt_pk_f16_f32 v154, v144, v145
	v_cvt_pk_f16_f32 v155, v146, v147
	v_cvt_pk_f16_f32 v152, v140, v141
	v_cvt_pk_f16_f32 v153, v142, v143
	v_cvt_pk_f16_f32 v150, v136, v137
	v_cvt_pk_f16_f32 v151, v138, v139
	v_cvt_pk_f16_f32 v148, v132, v133
	v_cvt_pk_f16_f32 v149, v134, v135
	v_cvt_pk_f16_f32 v146, v248, v249
	v_cvt_pk_f16_f32 v147, v250, v251
	v_cvt_pk_f16_f32 v144, v128, v129
	v_cvt_pk_f16_f32 v145, v130, v131
	v_cvt_pk_f16_f32 v142, v44, v45
	v_cvt_pk_f16_f32 v143, v46, v47
	v_cvt_pk_f16_f32 v140, v16, v17
	v_cvt_pk_f16_f32 v141, v18, v19
	v_cvt_pk_f16_f32 v138, v20, v21
	v_cvt_pk_f16_f32 v139, v22, v23
	v_cvt_pk_f16_f32 v136, v24, v25
	v_cvt_pk_f16_f32 v137, v26, v27
	v_cvt_pk_f16_f32 v134, v28, v29
	v_cvt_pk_f16_f32 v135, v30, v31
	v_cvt_pk_f16_f32 v132, v32, v33
	v_cvt_pk_f16_f32 v133, v34, v35
	v_cvt_pk_f16_f32 v46, v36, v37
	v_cvt_pk_f16_f32 v47, v38, v39
	v_cvt_pk_f16_f32 v44, v40, v41
	v_cvt_pk_f16_f32 v45, v42, v43
	v_cvt_pk_f16_f32 v42, v12, v13
	v_cvt_pk_f16_f32 v43, v14, v15
	v_cvt_pk_f16_f32 v40, v48, v49
	v_cvt_pk_f16_f32 v41, v50, v51
	v_cvt_pk_f16_f32 v38, v52, v53
	v_cvt_pk_f16_f32 v39, v54, v55
	v_cvt_pk_f16_f32 v36, v56, v57
	v_cvt_pk_f16_f32 v37, v58, v59
	v_cvt_pk_f16_f32 v34, v60, v61
	v_cvt_pk_f16_f32 v35, v62, v63
	v_cvt_pk_f16_f32 v32, v64, v65
	v_cvt_pk_f16_f32 v33, v66, v67
	v_cvt_pk_f16_f32 v30, v68, v69
	v_cvt_pk_f16_f32 v31, v70, v71
	v_cvt_pk_f16_f32 v28, v72, v73
	v_cvt_pk_f16_f32 v29, v74, v75
	v_cvt_pk_f16_f32 v26, v76, v77
	v_cvt_pk_f16_f32 v27, v78, v79
	v_cvt_pk_f16_f32 v24, v80, v81
	v_cvt_pk_f16_f32 v25, v82, v83
	v_cvt_pk_f16_f32 v22, v84, v85
	v_cvt_pk_f16_f32 v23, v86, v87
	v_cvt_pk_f16_f32 v20, v88, v89
	v_cvt_pk_f16_f32 v21, v90, v91
	v_cvt_pk_f16_f32 v18, v92, v93
	v_cvt_pk_f16_f32 v19, v94, v95
	v_cvt_pk_f16_f32 v16, v96, v97
	v_cvt_pk_f16_f32 v17, v98, v99
	v_cvt_pk_f16_f32 v14, v100, v101
	v_cvt_pk_f16_f32 v15, v102, v103
	v_cvt_pk_f16_f32 v12, v104, v105
	v_cvt_pk_f16_f32 v13, v106, v107
	v_cvt_pk_f16_f32 v72, v108, v109
	v_cvt_pk_f16_f32 v73, v110, v111
	v_cvt_pk_f16_f32 v70, v112, v113
	v_cvt_pk_f16_f32 v71, v114, v115
	v_cvt_pk_f16_f32 v52, v116, v117
	v_cvt_pk_f16_f32 v53, v118, v119
	v_cvt_pk_f16_f32 v50, v120, v121
	v_cvt_pk_f16_f32 v51, v122, v123
	v_cvt_pk_f16_f32 v48, v8, v9
	v_cvt_pk_f16_f32 v49, v10, v11
	v_cvt_pk_f16_f32 v0, v4, v5
	v_cvt_pk_f16_f32 v1, v6, v7
	s_lshl_b32 s3, s3, 3
	v_accvgpr_read_b32 v2, a209
	v_and_or_b32 v2, s3, 16, v2
	v_or_b32_e32 v3, 32, v2
	v_or_b32_e32 v4, 64, v2
	v_or_b32_e32 v5, 0x60, v2
	v_or_b32_e32 v6, 0x80, v2
	v_or_b32_e32 v7, 0xa0, v2
	v_or_b32_e32 v54, 0xc0, v2
	v_or_b32_e32 v55, 0xe0, v2
	v_accvgpr_read_b32 v56, a208
	v_lshrrev_b32_e32 v56, 2, v56
	v_and_b32_e32 v56, 28, v56
	v_or_b32_e32 v57, 32, v56
	v_or_b32_e32 v58, 64, v56
	v_or_b32_e32 v59, 0x60, v56
	v_or_b32_e32 v60, 0x80, v56
	v_or_b32_e32 v130, 0xa0, v56
	v_or_b32_e32 v129, 0xc0, v56
	v_or_b32_e32 v128, 0xe0, v56
	s_mul_i32 s6, s0, s4
	s_ashr_i32 s7, s6, 31
	s_lshl_b64 s[6:7], s[6:7], 1
	s_add_u32 s0, s12, s6
	s_addc_u32 s5, s13, s7
	s_ashr_i32 s3, s2, 31
	s_lshl_b64 s[2:3], s[2:3], 1
	s_add_u32 s0, s0, s2
	s_addc_u32 s33, s5, s3
	v_mul_lo_u32 v61, v2, s4
	v_mul_lo_u32 v62, v3, s4
	v_mul_lo_u32 v63, v4, s4
	v_mul_lo_u32 v64, v5, s4
	v_mul_lo_u32 v65, v6, s4
	v_mul_lo_u32 v66, v7, s4
	v_mul_lo_u32 v67, v54, s4
	v_mul_lo_u32 v68, v55, s4
	v_cmp_gt_i32_e32 vcc, s14, v2
	v_cmp_gt_i32_e64 s[12:13], s14, v3
	v_cmp_gt_i32_e64 s[16:17], s14, v4
	v_cmp_gt_i32_e64 s[18:19], s14, v5
	v_cmp_gt_i32_e64 s[20:21], s14, v6
	v_cmp_gt_i32_e64 s[22:23], s14, v7
	v_cmp_gt_i32_e64 s[26:27], s14, v54
	v_cmp_gt_i32_e64 s[28:29], s14, v55
	v_cmp_gt_i32_e64 s[2:3], s1, v56
	v_cmp_gt_i32_e64 s[14:15], s1, v57
	v_cmp_gt_i32_e64 s[4:5], s1, v58
	v_cmp_gt_i32_e64 s[6:7], s1, v59
	v_cmp_gt_i32_e64 s[8:9], s1, v60
	v_cmp_gt_i32_e64 s[10:11], s1, v130
	v_cmp_gt_i32_e64 s[44:45], s1, v129
	v_cmp_gt_i32_e64 s[24:25], s1, v128
	s_and_b64 s[30:31], vcc, s[2:3]
	s_and_b64 s[34:35], vcc, s[14:15]
	s_and_b64 s[36:37], vcc, s[4:5]
	s_and_b64 s[38:39], vcc, s[6:7]
	s_and_b64 s[40:41], vcc, s[8:9]
	s_and_b64 s[42:43], vcc, s[10:11]
	s_and_b64 s[46:47], vcc, s[44:45]
	s_and_b64 vcc, vcc, s[24:25]
	s_and_b64 s[48:49], s[12:13], s[2:3]
	s_and_b64 s[50:51], s[12:13], s[14:15]
	s_and_b64 s[52:53], s[12:13], s[4:5]
	s_and_b64 s[54:55], s[12:13], s[6:7]
	s_and_b64 s[56:57], s[12:13], s[8:9]
	s_and_b64 s[58:59], s[12:13], s[10:11]
	s_and_b64 s[98:99], s[12:13], s[44:45]
	s_and_b64 s[96:97], s[12:13], s[24:25]
	s_and_b64 s[94:95], s[16:17], s[2:3]
	s_and_b64 s[92:93], s[16:17], s[14:15]
	s_and_b64 s[90:91], s[16:17], s[4:5]
	s_and_b64 s[88:89], s[16:17], s[6:7]
	s_and_b64 s[86:87], s[16:17], s[8:9]
	s_and_b64 s[84:85], s[16:17], s[10:11]
	s_and_b64 s[82:83], s[16:17], s[44:45]
	s_and_b64 s[80:81], s[16:17], s[24:25]
	s_and_b64 s[78:79], s[18:19], s[2:3]
	s_and_b64 s[76:77], s[18:19], s[14:15]
	s_and_b64 s[74:75], s[18:19], s[4:5]
	s_and_b64 s[72:73], s[18:19], s[6:7]
	s_and_b64 s[70:71], s[18:19], s[8:9]
	s_and_b64 s[68:69], s[18:19], s[10:11]
	s_and_b64 s[66:67], s[18:19], s[44:45]
	s_and_b64 s[64:65], s[18:19], s[24:25]
	s_and_b64 s[62:63], s[20:21], s[2:3]
	s_and_b64 s[60:61], s[20:21], s[14:15]
	s_and_b64 s[12:13], s[20:21], s[4:5]
	s_and_b64 s[16:17], s[20:21], s[6:7]
	s_and_b64 s[18:19], s[20:21], s[8:9]
                                        ; implicit-def: $vgpr255 : SGPR spill to VGPR lane
	v_writelane_b32 v255, s18, 0
	s_nop 1
	v_writelane_b32 v255, s19, 1
	s_and_b64 s[18:19], s[20:21], s[10:11]
	v_writelane_b32 v255, s18, 2
	s_nop 1
	v_writelane_b32 v255, s19, 3
	s_and_b64 s[18:19], s[20:21], s[44:45]
	v_writelane_b32 v255, s18, 4
	s_nop 1
	v_writelane_b32 v255, s19, 5
	s_and_b64 s[20:21], s[20:21], s[24:25]
	v_writelane_b32 v255, s20, 6
	s_nop 1
	v_writelane_b32 v255, s21, 7
	s_and_b64 s[20:21], s[22:23], s[2:3]
	v_writelane_b32 v255, s20, 8
	s_nop 1
	v_writelane_b32 v255, s21, 9
	s_and_b64 s[20:21], s[22:23], s[14:15]
	v_writelane_b32 v255, s20, 10
	s_nop 1
	v_writelane_b32 v255, s21, 11
	s_and_b64 s[20:21], s[22:23], s[4:5]
	v_writelane_b32 v255, s20, 12
	s_nop 1
	v_writelane_b32 v255, s21, 13
	s_and_b64 s[20:21], s[22:23], s[6:7]
	v_writelane_b32 v255, s20, 14
	s_nop 1
	v_writelane_b32 v255, s21, 15
	s_and_b64 s[20:21], s[22:23], s[8:9]
	v_writelane_b32 v255, s20, 16
	s_nop 1
	v_writelane_b32 v255, s21, 17
	s_and_b64 s[20:21], s[22:23], s[10:11]
	v_writelane_b32 v255, s20, 18
	s_nop 1
	v_writelane_b32 v255, s21, 19
	s_and_b64 s[20:21], s[22:23], s[44:45]
	v_writelane_b32 v255, s20, 20
	s_nop 1
	v_writelane_b32 v255, s21, 21
	s_and_b64 s[22:23], s[22:23], s[24:25]
	v_writelane_b32 v255, s22, 22
	s_nop 1
	v_writelane_b32 v255, s23, 23
	s_and_b64 s[22:23], s[26:27], s[2:3]
	v_writelane_b32 v255, s22, 24
	s_nop 1
	v_writelane_b32 v255, s23, 25
	s_and_b64 s[22:23], s[26:27], s[14:15]
	v_writelane_b32 v255, s22, 26
	s_nop 1
	v_writelane_b32 v255, s23, 27
	s_and_b64 s[22:23], s[26:27], s[4:5]
	v_writelane_b32 v255, s22, 28
	s_nop 1
	v_writelane_b32 v255, s23, 29
	s_and_b64 s[22:23], s[26:27], s[6:7]
	v_writelane_b32 v255, s22, 30
	s_nop 1
	v_writelane_b32 v255, s23, 31
	s_and_b64 s[22:23], s[26:27], s[8:9]
	v_writelane_b32 v255, s22, 32
	s_nop 1
	v_writelane_b32 v255, s23, 33
	s_and_b64 s[22:23], s[26:27], s[10:11]
	v_writelane_b32 v255, s22, 34
	s_nop 1
	v_writelane_b32 v255, s23, 35
	s_and_b64 s[22:23], s[26:27], s[44:45]
	v_writelane_b32 v255, s22, 36
	s_nop 1
	v_writelane_b32 v255, s23, 37
	s_and_b64 s[26:27], s[26:27], s[24:25]
	v_writelane_b32 v255, s26, 38
	s_nop 1
	v_writelane_b32 v255, s27, 39
	s_and_b64 s[2:3], s[28:29], s[2:3]
	v_writelane_b32 v255, s2, 40
	s_nop 1
	v_writelane_b32 v255, s3, 41
	s_and_b64 s[2:3], s[28:29], s[14:15]
	v_writelane_b32 v255, s2, 42
	s_nop 1
	v_writelane_b32 v255, s3, 43
	s_and_b64 s[2:3], s[28:29], s[4:5]
	v_writelane_b32 v255, s2, 44
	s_nop 1
	v_writelane_b32 v255, s3, 45
	s_and_b64 s[26:27], s[28:29], s[6:7]
	s_and_b64 s[14:15], s[28:29], s[8:9]
	s_and_b64 s[8:9], s[28:29], s[10:11]
	s_and_b64 s[6:7], s[28:29], s[44:45]
	s_and_b64 s[4:5], s[28:29], s[24:25]
	s_and_b32 s1, s33, 0xffff
	s_mov_b32 s3, 0x27000
	s_mov_b32 s2, 0x7ffffffe
	v_add_lshl_u32 v2, v56, v61, 1
	v_bfrev_b32_e32 v3, 1
	v_cndmask_b32_e64 v2, v3, v2, s[30:31]
	buffer_store_dwordx2 v[126:127], v2, s[0:3], 0 offen
	v_add_lshl_u32 v2, v57, v61, 1
	v_cndmask_b32_e64 v2, v3, v2, s[34:35]
	buffer_store_dwordx2 v[124:125], v2, s[0:3], 0 offen
	v_add_lshl_u32 v2, v58, v61, 1
	v_cndmask_b32_e64 v2, v3, v2, s[36:37]
	buffer_store_dwordx2 v[252:253], v2, s[0:3], 0 offen
	v_add_lshl_u32 v2, v59, v61, 1
	v_cndmask_b32_e64 v2, v3, v2, s[38:39]
	buffer_store_dwordx2 v[244:245], v2, s[0:3], 0 offen
	v_add_lshl_u32 v2, v60, v61, 1
	v_cndmask_b32_e64 v2, v3, v2, s[40:41]
	buffer_store_dwordx2 v[240:241], v2, s[0:3], 0 offen
	v_add_lshl_u32 v2, v130, v61, 1
	v_cndmask_b32_e64 v2, v3, v2, s[42:43]
	buffer_store_dwordx2 v[236:237], v2, s[0:3], 0 offen
	v_add_lshl_u32 v2, v129, v61, 1
	v_cndmask_b32_e64 v2, v3, v2, s[46:47]
	buffer_store_dwordx2 v[232:233], v2, s[0:3], 0 offen
	v_add_lshl_u32 v2, v128, v61, 1
	v_cndmask_b32_e32 v2, v3, v2, vcc
	buffer_store_dwordx2 v[228:229], v2, s[0:3], 0 offen
	v_add_lshl_u32 v2, v62, v56, 1
	v_cndmask_b32_e64 v2, v3, v2, s[48:49]
	buffer_store_dwordx2 v[224:225], v2, s[0:3], 0 offen
	v_add_lshl_u32 v2, v57, v62, 1
	v_cndmask_b32_e64 v2, v3, v2, s[50:51]
	buffer_store_dwordx2 v[220:221], v2, s[0:3], 0 offen
	v_add_lshl_u32 v2, v58, v62, 1
	v_cndmask_b32_e64 v2, v3, v2, s[52:53]
	buffer_store_dwordx2 v[216:217], v2, s[0:3], 0 offen
	v_add_lshl_u32 v2, v59, v62, 1
	v_cndmask_b32_e64 v2, v3, v2, s[54:55]
	buffer_store_dwordx2 v[212:213], v2, s[0:3], 0 offen
	v_add_lshl_u32 v2, v60, v62, 1
	v_cndmask_b32_e64 v2, v3, v2, s[56:57]
	buffer_store_dwordx2 v[208:209], v2, s[0:3], 0 offen
	v_add_lshl_u32 v2, v130, v62, 1
	v_cndmask_b32_e64 v2, v3, v2, s[58:59]
	buffer_store_dwordx2 v[204:205], v2, s[0:3], 0 offen
	v_add_lshl_u32 v2, v129, v62, 1
	v_cndmask_b32_e64 v2, v3, v2, s[98:99]
	buffer_store_dwordx2 v[200:201], v2, s[0:3], 0 offen
	v_add_lshl_u32 v2, v128, v62, 1
	v_cndmask_b32_e64 v2, v3, v2, s[96:97]
	buffer_store_dwordx2 v[196:197], v2, s[0:3], 0 offen
	v_add_lshl_u32 v2, v63, v56, 1
	v_cndmask_b32_e64 v2, v3, v2, s[94:95]
	buffer_store_dwordx2 v[190:191], v2, s[0:3], 0 offen
	v_add_lshl_u32 v2, v57, v63, 1
	v_cndmask_b32_e64 v2, v3, v2, s[92:93]
	buffer_store_dwordx2 v[188:189], v2, s[0:3], 0 offen
	v_add_lshl_u32 v2, v58, v63, 1
	v_cndmask_b32_e64 v2, v3, v2, s[90:91]
	buffer_store_dwordx2 v[184:185], v2, s[0:3], 0 offen
	v_add_lshl_u32 v2, v59, v63, 1
	v_cndmask_b32_e64 v2, v3, v2, s[88:89]
	buffer_store_dwordx2 v[180:181], v2, s[0:3], 0 offen
	v_add_lshl_u32 v2, v60, v63, 1
	v_cndmask_b32_e64 v2, v3, v2, s[86:87]
	buffer_store_dwordx2 v[176:177], v2, s[0:3], 0 offen
	v_add_lshl_u32 v2, v130, v63, 1
	v_cndmask_b32_e64 v2, v3, v2, s[84:85]
	buffer_store_dwordx2 v[172:173], v2, s[0:3], 0 offen
	v_add_lshl_u32 v2, v129, v63, 1
	v_cndmask_b32_e64 v2, v3, v2, s[82:83]
	buffer_store_dwordx2 v[168:169], v2, s[0:3], 0 offen
	v_add_lshl_u32 v2, v128, v63, 1
	v_cndmask_b32_e64 v2, v3, v2, s[80:81]
	buffer_store_dwordx2 v[164:165], v2, s[0:3], 0 offen
	v_add_lshl_u32 v2, v64, v56, 1
	v_cndmask_b32_e64 v2, v3, v2, s[78:79]
	buffer_store_dwordx2 v[162:163], v2, s[0:3], 0 offen
	v_add_lshl_u32 v2, v57, v64, 1
	v_cndmask_b32_e64 v2, v3, v2, s[76:77]
	buffer_store_dwordx2 v[160:161], v2, s[0:3], 0 offen
	v_add_lshl_u32 v2, v58, v64, 1
	v_cndmask_b32_e64 v2, v3, v2, s[74:75]
	buffer_store_dwordx2 v[158:159], v2, s[0:3], 0 offen
	v_add_lshl_u32 v2, v59, v64, 1
	v_cndmask_b32_e64 v2, v3, v2, s[72:73]
	buffer_store_dwordx2 v[156:157], v2, s[0:3], 0 offen
	v_add_lshl_u32 v2, v60, v64, 1
	v_cndmask_b32_e64 v2, v3, v2, s[70:71]
	buffer_store_dwordx2 v[154:155], v2, s[0:3], 0 offen
	v_add_lshl_u32 v2, v130, v64, 1
	v_cndmask_b32_e64 v2, v3, v2, s[68:69]
	buffer_store_dwordx2 v[152:153], v2, s[0:3], 0 offen
	v_add_lshl_u32 v2, v129, v64, 1
	v_cndmask_b32_e64 v2, v3, v2, s[66:67]
	buffer_store_dwordx2 v[150:151], v2, s[0:3], 0 offen
	v_add_lshl_u32 v2, v128, v64, 1
	v_cndmask_b32_e64 v2, v3, v2, s[64:65]
	buffer_store_dwordx2 v[148:149], v2, s[0:3], 0 offen
	v_add_lshl_u32 v2, v65, v56, 1
	v_cndmask_b32_e64 v2, v3, v2, s[62:63]
	buffer_store_dwordx2 v[146:147], v2, s[0:3], 0 offen
	v_add_lshl_u32 v2, v57, v65, 1
	v_cndmask_b32_e64 v2, v3, v2, s[60:61]
	buffer_store_dwordx2 v[144:145], v2, s[0:3], 0 offen
	v_add_lshl_u32 v2, v58, v65, 1
	v_cndmask_b32_e64 v2, v3, v2, s[12:13]
	buffer_store_dwordx2 v[142:143], v2, s[0:3], 0 offen
	v_add_lshl_u32 v2, v59, v65, 1
	v_cndmask_b32_e64 v2, v3, v2, s[16:17]
	buffer_store_dwordx2 v[140:141], v2, s[0:3], 0 offen
	v_add_lshl_u32 v2, v60, v65, 1
	v_readlane_b32 s10, v255, 0
	v_readlane_b32 s11, v255, 1
	s_nop 1
	v_cndmask_b32_e64 v2, v3, v2, s[10:11]
	buffer_store_dwordx2 v[138:139], v2, s[0:3], 0 offen
	v_add_lshl_u32 v2, v130, v65, 1
	v_readlane_b32 s10, v255, 2
	v_readlane_b32 s11, v255, 3
	s_nop 1
	v_cndmask_b32_e64 v2, v3, v2, s[10:11]
	buffer_store_dwordx2 v[136:137], v2, s[0:3], 0 offen
	v_add_lshl_u32 v2, v129, v65, 1
	v_readlane_b32 s10, v255, 4
	v_readlane_b32 s11, v255, 5
	s_nop 1
	v_cndmask_b32_e64 v2, v3, v2, s[10:11]
	buffer_store_dwordx2 v[134:135], v2, s[0:3], 0 offen
	v_add_lshl_u32 v2, v128, v65, 1
	v_readlane_b32 s10, v255, 6
	v_readlane_b32 s11, v255, 7
	s_nop 1
	v_cndmask_b32_e64 v2, v3, v2, s[10:11]
	buffer_store_dwordx2 v[132:133], v2, s[0:3], 0 offen
	v_add_lshl_u32 v2, v66, v56, 1
	v_readlane_b32 s10, v255, 8
	v_readlane_b32 s11, v255, 9
	s_nop 1
	v_cndmask_b32_e64 v2, v3, v2, s[10:11]
	buffer_store_dwordx2 v[46:47], v2, s[0:3], 0 offen
	v_add_lshl_u32 v2, v57, v66, 1
	v_readlane_b32 s10, v255, 10
	v_readlane_b32 s11, v255, 11
	s_nop 1
	v_cndmask_b32_e64 v2, v3, v2, s[10:11]
	buffer_store_dwordx2 v[44:45], v2, s[0:3], 0 offen
	v_add_lshl_u32 v2, v58, v66, 1
	v_readlane_b32 s10, v255, 12
	v_readlane_b32 s11, v255, 13
	s_nop 1
	v_cndmask_b32_e64 v2, v3, v2, s[10:11]
	buffer_store_dwordx2 v[42:43], v2, s[0:3], 0 offen
	v_add_lshl_u32 v2, v59, v66, 1
	v_readlane_b32 s10, v255, 14
	v_readlane_b32 s11, v255, 15
	s_nop 1
	v_cndmask_b32_e64 v2, v3, v2, s[10:11]
	buffer_store_dwordx2 v[40:41], v2, s[0:3], 0 offen
	v_add_lshl_u32 v2, v60, v66, 1
	v_readlane_b32 s10, v255, 16
	v_readlane_b32 s11, v255, 17
	s_nop 1
	v_cndmask_b32_e64 v2, v3, v2, s[10:11]
	buffer_store_dwordx2 v[38:39], v2, s[0:3], 0 offen
	v_add_lshl_u32 v2, v130, v66, 1
	v_readlane_b32 s10, v255, 18
	v_readlane_b32 s11, v255, 19
	s_nop 1
	v_cndmask_b32_e64 v2, v3, v2, s[10:11]
	buffer_store_dwordx2 v[36:37], v2, s[0:3], 0 offen
	v_add_lshl_u32 v2, v129, v66, 1
	v_readlane_b32 s10, v255, 20
	v_readlane_b32 s11, v255, 21
	s_nop 1
	v_cndmask_b32_e64 v2, v3, v2, s[10:11]
	buffer_store_dwordx2 v[34:35], v2, s[0:3], 0 offen
	v_add_lshl_u32 v2, v128, v66, 1
	v_readlane_b32 s10, v255, 22
	v_readlane_b32 s11, v255, 23
	s_nop 1
	v_cndmask_b32_e64 v2, v3, v2, s[10:11]
	buffer_store_dwordx2 v[32:33], v2, s[0:3], 0 offen
	v_add_lshl_u32 v2, v67, v56, 1
	v_readlane_b32 s10, v255, 24
	v_readlane_b32 s11, v255, 25
	s_nop 1
	v_cndmask_b32_e64 v2, v3, v2, s[10:11]
	buffer_store_dwordx2 v[30:31], v2, s[0:3], 0 offen
	v_add_lshl_u32 v2, v57, v67, 1
	v_readlane_b32 s10, v255, 26
	v_readlane_b32 s11, v255, 27
	s_nop 1
	v_cndmask_b32_e64 v2, v3, v2, s[10:11]
	buffer_store_dwordx2 v[28:29], v2, s[0:3], 0 offen
	v_add_lshl_u32 v2, v58, v67, 1
	v_readlane_b32 s10, v255, 28
	v_readlane_b32 s11, v255, 29
	s_nop 1
	v_cndmask_b32_e64 v2, v3, v2, s[10:11]
	buffer_store_dwordx2 v[26:27], v2, s[0:3], 0 offen
	v_add_lshl_u32 v2, v59, v67, 1
	v_readlane_b32 s10, v255, 30
	v_readlane_b32 s11, v255, 31
	s_nop 1
	v_cndmask_b32_e64 v2, v3, v2, s[10:11]
	buffer_store_dwordx2 v[24:25], v2, s[0:3], 0 offen
	v_add_lshl_u32 v2, v60, v67, 1
	v_readlane_b32 s10, v255, 32
	v_readlane_b32 s11, v255, 33
	s_nop 1
	v_cndmask_b32_e64 v2, v3, v2, s[10:11]
	buffer_store_dwordx2 v[22:23], v2, s[0:3], 0 offen
	v_add_lshl_u32 v2, v130, v67, 1
	v_readlane_b32 s10, v255, 34
	v_readlane_b32 s11, v255, 35
	s_nop 1
	v_cndmask_b32_e64 v2, v3, v2, s[10:11]
	buffer_store_dwordx2 v[20:21], v2, s[0:3], 0 offen
	v_add_lshl_u32 v2, v129, v67, 1
	v_readlane_b32 s10, v255, 36
	v_readlane_b32 s11, v255, 37
	s_nop 1
	v_cndmask_b32_e64 v2, v3, v2, s[10:11]
	buffer_store_dwordx2 v[18:19], v2, s[0:3], 0 offen
	v_add_lshl_u32 v2, v128, v67, 1
	v_readlane_b32 s10, v255, 38
	v_readlane_b32 s11, v255, 39
	s_nop 1
	v_cndmask_b32_e64 v2, v3, v2, s[10:11]
	buffer_store_dwordx2 v[16:17], v2, s[0:3], 0 offen
	v_add_lshl_u32 v2, v68, v56, 1
	v_readlane_b32 s10, v255, 40
	v_readlane_b32 s11, v255, 41
	s_nop 1
	v_cndmask_b32_e64 v2, v3, v2, s[10:11]
	buffer_store_dwordx2 v[14:15], v2, s[0:3], 0 offen
	v_add_lshl_u32 v2, v57, v68, 1
	v_readlane_b32 s10, v255, 42
	v_readlane_b32 s11, v255, 43
	s_nop 1
	v_cndmask_b32_e64 v2, v3, v2, s[10:11]
	buffer_store_dwordx2 v[12:13], v2, s[0:3], 0 offen
	v_add_lshl_u32 v2, v58, v68, 1
	v_readlane_b32 s10, v255, 44
	v_readlane_b32 s11, v255, 45
	s_nop 1
	v_cndmask_b32_e64 v2, v3, v2, s[10:11]
	buffer_store_dwordx2 v[72:73], v2, s[0:3], 0 offen
	v_add_lshl_u32 v2, v59, v68, 1
	v_cndmask_b32_e64 v2, v3, v2, s[26:27]
	buffer_store_dwordx2 v[70:71], v2, s[0:3], 0 offen
	v_add_lshl_u32 v2, v60, v68, 1
	v_cndmask_b32_e64 v2, v3, v2, s[14:15]
	buffer_store_dwordx2 v[52:53], v2, s[0:3], 0 offen
	v_add_lshl_u32 v2, v130, v68, 1
	v_cndmask_b32_e64 v2, v3, v2, s[8:9]
	buffer_store_dwordx2 v[50:51], v2, s[0:3], 0 offen
	v_add_lshl_u32 v2, v129, v68, 1
	v_cndmask_b32_e64 v2, v3, v2, s[6:7]
	buffer_store_dwordx2 v[48:49], v2, s[0:3], 0 offen
	v_add_lshl_u32 v2, v128, v68, 1
	v_cndmask_b32_e64 v2, v3, v2, s[4:5]
	buffer_store_dwordx2 v[0:1], v2, s[0:3], 0 offen
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
		.amdhsa_next_free_vgpr 510
		.amdhsa_next_free_sgpr 100
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
	.set v5_local_prefetch.num_agpr, 254
	.set v5_local_prefetch.numbered_sgpr, 100
	.set v5_local_prefetch.num_named_barrier, 0
	.set v5_local_prefetch.private_seg_size, 0
	.set v5_local_prefetch.uses_vcc, 1
	.set v5_local_prefetch.uses_flat_scratch, 0
	.set v5_local_prefetch.has_dyn_sized_stack, 0
	.set v5_local_prefetch.has_recursion, 0
	.set v5_local_prefetch.has_indirect_call, 0
	.section	.AMDGPU.csdata,"",@progbits
; Kernel info:
; codeLenInByte = 17232
; TotalNumSgprs: 106
; NumVgprs: 256
; NumAgprs: 254
; TotalNumVgprs: 510
; ScratchSize: 0
; MemoryBound: 0
; FloatMode: 240
; IeeeMode: 1
; LDSByteSize: 0 bytes/workgroup (compile time only)
; SGPRBlocks: 13
; VGPRBlocks: 63
; NumSGPRsForWavesPerEU: 106
; NumVGPRsForWavesPerEU: 510
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
	.quad	.Ltmp2                          ; DW_AT_low_pc
	.long	.Ltmp3-.Ltmp2                   ; DW_AT_high_pc
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
	.quad	.Ltmp4-.Lfunc_begin0
	.quad	.Ltmp5-.Lfunc_begin0
	.quad	.Ltmp6-.Lfunc_begin0
	.quad	.Ltmp7-.Lfunc_begin0
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
    .sgpr_count:     106
    .sgpr_spill_count: 46
    .symbol:         v5_local_prefetch.kd
    .uniform_work_group_size: 1
    .uses_dynamic_stack: false
    .vgpr_count:     510
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
