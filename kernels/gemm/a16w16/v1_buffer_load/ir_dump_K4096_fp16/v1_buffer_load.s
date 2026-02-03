	.amdgcn_target "amdgcn-amd-amdhsa--gfx950"
	.amdhsa_code_object_version 5
	.text
	.globl	v1_buffer_load                  ; -- Begin function v1_buffer_load
	.p2align	8
	.type	v1_buffer_load,@function
v1_buffer_load:                         ; @v1_buffer_load
.Lfunc_begin0:
	.cfi_sections .debug_frame
	.cfi_startproc
; %bb.6:
	.file	1 "/root/gfx9-gluon-tutorials/kernels/gemm/a16w16/v1_buffer_load" "matmul_kernel.py"
	s_load_dwordx2 s[2:3], s[0:1], 0x0
	s_load_dwordx8 s[4:11], s[0:1], 0x8
	s_load_dwordx4 s[12:15], s[0:1], 0x28
	s_waitcnt lgkmcnt(0)
	s_branch .LBB0_0
	.p2align	8
; %bb.7:
.LBB0_0:
	.file	2 "/root/OAI-triton/python/triton/language" "standard.py"
	s_add_i32 s0, s9, 0xff
	s_ashr_i32 s1, s0, 31
	s_lshr_b32 s1, s1, 24
	s_add_i32 s0, s0, s1
	s_ashr_i32 s0, s0, 8
	s_abs_i32 s1, s0
	v_cvt_f32_u32_e32 v1, s1
	s_sub_i32 s19, 0, s1
	s_abs_i32 s18, s16
	v_readfirstlane_b32 s15, v0
	v_rcp_iflag_f32_e32 v1, v1
	s_xor_b32 s14, s16, s0
	s_and_b32 s17, s15, 0xc0
	s_ashr_i32 s14, s14, 31
	v_mul_f32_e32 v1, 0x4f7ffffe, v1
	v_cvt_u32_f32_e32 v1, v1
	v_and_or_b32 v50, v0, 63, s17
	v_readfirstlane_b32 s20, v1
	s_mul_i32 s19, s19, s20
	s_mul_hi_u32 s19, s20, s19
	s_add_i32 s20, s20, s19
	s_mul_hi_u32 s19, s18, s20
	s_mul_i32 s20, s19, s1
	s_sub_i32 s18, s18, s20
	s_add_i32 s20, s19, 1
	s_sub_i32 s21, s18, s1
	s_cmp_ge_u32 s18, s1
	s_cselect_b32 s19, s20, s19
	s_cselect_b32 s18, s21, s18
	s_add_i32 s20, s19, 1
	s_cmp_ge_u32 s18, s1
	s_cselect_b32 s1, s20, s19
	s_xor_b32 s1, s1, s14
	s_sub_i32 s1, s1, s14
	s_mul_i32 s0, s1, s0
	s_sub_i32 s0, s16, s0
	s_lshl_b32 s16, s1, 8
	s_lshl_b32 s14, s0, 8
	s_add_i32 s0, s10, 63
	s_cmp_lt_i32 s0, 64
	s_cbranch_scc1 .LBB0_4
; %bb.1:                                ; %.lr.ph
	v_lshlrev_b32_e32 v17, 3, v0
	v_lshrrev_b32_e32 v1, 3, v50
	v_and_b32_e32 v18, 56, v17
	v_mul_lo_u32 v16, v1, s12
	s_ashr_i32 s1, s0, 31
	s_lshr_b32 s1, s1, 26
	s_add_i32 s0, s0, s1
	s_ashr_i32 s17, s0, 6
	s_mul_i32 s0, s14, s12
	s_ashr_i32 s1, s0, 31
	s_lshl_b64 s[0:1], s[0:1], 1
	s_add_u32 s4, s4, s0
	s_addc_u32 s5, s5, s1
	v_or_b32_e32 v2, 0xe0, v1
	v_mul_lo_u32 v3, v2, s12
	v_mul_lo_u32 v2, v2, s11
	s_mul_i32 s0, s16, s11
	s_ashr_i32 s1, s0, 31
	s_lshl_b64 s[0:1], s[0:1], 1
	v_or_b32_e32 v4, 0xc0, v1
	v_mul_lo_u32 v5, v4, s12
	v_mul_lo_u32 v4, v4, s11
	v_or_b32_e32 v6, 0xa0, v1
	v_mul_lo_u32 v7, v6, s12
	v_mul_lo_u32 v6, v6, s11
	v_or_b32_e32 v8, 0x80, v1
	v_mul_lo_u32 v9, v8, s12
	v_mul_lo_u32 v8, v8, s11
	v_or_b32_e32 v10, 0x60, v1
	v_mul_lo_u32 v11, v10, s12
	v_mul_lo_u32 v10, v10, s11
	v_or_b32_e32 v12, 64, v1
	v_mul_lo_u32 v13, v12, s12
	v_mul_lo_u32 v12, v12, s11
	v_or_b32_e32 v14, 32, v1
	v_mul_lo_u32 v15, v14, s12
	v_mul_lo_u32 v14, v14, s11
	s_movk_i32 s18, 0xe0
	v_mul_lo_u32 v1, v1, s11
	v_add_lshl_u32 v1, v1, v18, 1
	v_accvgpr_write_b32 a107, v1
	v_add_lshl_u32 v1, v14, v18, 1
	v_accvgpr_write_b32 a108, v1
	v_add_lshl_u32 v1, v12, v18, 1
	v_accvgpr_write_b32 a109, v1
	v_add_lshl_u32 v1, v10, v18, 1
	v_accvgpr_write_b32 a130, v1
	v_add_lshl_u32 v1, v8, v18, 1
	v_accvgpr_write_b32 a131, v1
	v_add_lshl_u32 v1, v6, v18, 1
	v_accvgpr_write_b32 a132, v1
	v_add_lshl_u32 v1, v4, v18, 1
	v_accvgpr_write_b32 a133, v1
	v_add_lshl_u32 v1, v2, v18, 1
	v_accvgpr_write_b32 a134, v1
	v_add_lshl_u32 v1, v16, v18, 1
	v_accvgpr_write_b32 a135, v1
	v_add_lshl_u32 v1, v15, v18, 1
	v_accvgpr_write_b32 a136, v1
	v_add_lshl_u32 v1, v13, v18, 1
	v_accvgpr_write_b32 a137, v1
	v_add_lshl_u32 v1, v11, v18, 1
	v_accvgpr_write_b32 a138, v1
	v_add_lshl_u32 v1, v9, v18, 1
	v_accvgpr_write_b32 a139, v1
	v_add_lshl_u32 v1, v7, v18, 1
	v_accvgpr_write_b32 a140, v1
	v_add_lshl_u32 v1, v5, v18, 1
	v_accvgpr_write_b32 a141, v1
	v_add_lshl_u32 v1, v3, v18, 1
	v_accvgpr_write_b32 a142, v1
	v_lshlrev_b32_e32 v1, 4, v0
	v_and_b32_e32 v1, 0x60, v1
	s_add_u32 s11, s2, s0
	s_movk_i32 s19, 0xa0
	v_and_b32_e32 v3, 0xe0, v50
	s_addc_u32 s12, s3, s1
	v_accvgpr_write_b32 a104, v0
	v_and_b32_e32 v7, 16, v0
	s_lshl_b32 s0, s15, 1
	s_and_b32 s0, s0, 0x80
	v_and_b32_e32 v11, 0x60, v17
	v_bitop3_b32 v9, v50, v11, s19 bitop3:0x6c
	v_accvgpr_write_b32 a106, v18
	v_accvgpr_write_b32 a105, v50
	v_lshlrev_b32_e32 v2, 3, v50
	v_and_b32_e32 v2, 0x300, v2
	v_bitop3_b32 v2, v2, v3, v1 bitop3:0x36
	v_lshl_add_u32 v3, v7, 7, 0
	v_lshl_or_b32 v7, v7, 6, s0
	s_movk_i32 s0, 0x120
	v_and_b32_e32 v4, 1, v0
	v_lshlrev_b32_e32 v5, 10, v4
	v_lshlrev_b32_e32 v4, 4, v4
	v_bitop3_b32 v1, v50, v1, s18 bitop3:0x6c
	v_lshlrev_b32_e32 v6, 1, v0
	v_and_b32_e32 v6, 16, v6
	v_add3_u32 v2, v3, v2, v6
	v_add_u32_e32 v2, v2, v5
	v_mov_b32_e32 v134, 0
	v_lshlrev_b32_e32 v8, 10, v0
	v_and_b32_e32 v8, 0x800, v8
	v_lshlrev_b32_e32 v3, 6, v0
	v_and_b32_e32 v3, 0x700, v3
	v_or3_b32 v3, v3, v9, v4
	v_or_b32_e32 v9, v3, v8
	v_bitop3_b32 v3, v3, 64, v8 bitop3:0x36
	v_lshlrev_b32_e32 v10, 7, v0
	v_and_b32_e32 v10, 0xb00, v10
	v_add_u32_e32 v10, 0, v10
	v_add3_u32 v1, v10, v1, v5
	v_add_u32_e32 v1, v1, v6
	v_accvgpr_write_b32 a160, v1
	v_add_u32_e32 v1, 0, v9
	v_accvgpr_write_b32 a161, v1
	v_add_u32_e32 v1, 0, v3
	v_mov_b32_e32 v135, 0
	v_mov_b32_e32 v136, 0
	v_bfe_i32 v10, v0, 5, 1
	v_bitop3_b32 v0, v10, v11, s0 bitop3:0x6c
	v_or3_b32 v0, v7, v0, v8
	v_or_b32_e32 v7, v0, v4
	v_mov_b32_e32 v137, 0
	v_mov_b32_e32 v154, 0
	v_bitop3_b32 v0, v0, 64, v4 bitop3:0x36
	v_mov_b32_e32 v155, 0
	v_mov_b32_e32 v156, 0
	v_mov_b32_e32 v157, 0
	v_mov_b32_e32 v22, 0
	v_mov_b32_e32 v23, 0
	v_mov_b32_e32 v24, 0
	v_mov_b32_e32 v25, 0
	v_accvgpr_write_b32 a118, 0
	v_accvgpr_write_b32 a119, 0
	v_accvgpr_write_b32 a120, 0
	v_accvgpr_write_b32 a121, 0
	v_accvgpr_write_b32 a122, 0
	v_accvgpr_write_b32 a123, 0
	v_accvgpr_write_b32 a124, 0
	v_accvgpr_write_b32 a125, 0
	v_accvgpr_write_b32 a126, 0
	v_accvgpr_write_b32 a127, 0
	v_accvgpr_write_b32 a128, 0
	v_accvgpr_write_b32 a129, 0
	v_mov_b32_e32 v18, 0
	v_mov_b32_e32 v19, 0
	v_mov_b32_e32 v20, 0
	v_mov_b32_e32 v21, 0
	v_mov_b32_e32 v10, 0
	v_mov_b32_e32 v11, 0
	v_mov_b32_e32 v12, 0
	v_mov_b32_e32 v13, 0
	v_accvgpr_write_b32 a100, 0
	v_accvgpr_write_b32 a101, 0
	v_accvgpr_write_b32 a102, 0
	v_accvgpr_write_b32 a103, 0
	v_accvgpr_write_b32 a96, 0
	v_accvgpr_write_b32 a97, 0
	v_accvgpr_write_b32 a98, 0
	v_accvgpr_write_b32 a99, 0
	v_mov_b32_e32 v250, 0
	v_mov_b32_e32 v251, 0
	v_mov_b32_e32 v252, 0
	v_mov_b32_e32 v253, 0
	v_mov_b32_e32 v246, 0
	v_mov_b32_e32 v247, 0
	v_mov_b32_e32 v248, 0
	v_mov_b32_e32 v249, 0
	v_mov_b32_e32 v242, 0
	v_mov_b32_e32 v243, 0
	v_mov_b32_e32 v244, 0
	v_mov_b32_e32 v245, 0
	v_mov_b32_e32 v238, 0
	v_mov_b32_e32 v239, 0
	v_mov_b32_e32 v240, 0
	v_mov_b32_e32 v241, 0
	v_mov_b32_e32 v234, 0
	v_mov_b32_e32 v235, 0
	v_mov_b32_e32 v236, 0
	v_mov_b32_e32 v237, 0
	v_mov_b32_e32 v230, 0
	v_mov_b32_e32 v231, 0
	v_mov_b32_e32 v232, 0
	v_mov_b32_e32 v233, 0
	v_mov_b32_e32 v226, 0
	v_mov_b32_e32 v227, 0
	v_mov_b32_e32 v228, 0
	v_mov_b32_e32 v229, 0
	v_mov_b32_e32 v222, 0
	v_mov_b32_e32 v223, 0
	v_mov_b32_e32 v224, 0
	v_mov_b32_e32 v225, 0
	v_mov_b32_e32 v218, 0
	v_mov_b32_e32 v219, 0
	v_mov_b32_e32 v220, 0
	v_mov_b32_e32 v221, 0
	v_mov_b32_e32 v214, 0
	v_mov_b32_e32 v215, 0
	v_mov_b32_e32 v216, 0
	v_mov_b32_e32 v217, 0
	v_mov_b32_e32 v210, 0
	v_mov_b32_e32 v211, 0
	v_mov_b32_e32 v212, 0
	v_mov_b32_e32 v213, 0
	v_mov_b32_e32 v206, 0
	v_mov_b32_e32 v207, 0
	v_mov_b32_e32 v208, 0
	v_mov_b32_e32 v209, 0
	v_mov_b32_e32 v202, 0
	v_mov_b32_e32 v203, 0
	v_mov_b32_e32 v204, 0
	v_mov_b32_e32 v205, 0
	v_accvgpr_write_b32 a252, 0
	v_accvgpr_write_b32 a253, 0
	v_accvgpr_write_b32 a254, 0
	v_accvgpr_write_b32 a255, 0
	v_accvgpr_write_b32 a248, 0
	v_accvgpr_write_b32 a249, 0
	v_accvgpr_write_b32 a250, 0
	v_accvgpr_write_b32 a251, 0
	v_accvgpr_write_b32 a244, 0
	v_accvgpr_write_b32 a245, 0
	v_accvgpr_write_b32 a246, 0
	v_accvgpr_write_b32 a247, 0
	v_accvgpr_write_b32 a240, 0
	v_accvgpr_write_b32 a241, 0
	v_accvgpr_write_b32 a242, 0
	v_accvgpr_write_b32 a243, 0
	v_accvgpr_write_b32 a236, 0
	v_accvgpr_write_b32 a237, 0
	v_accvgpr_write_b32 a238, 0
	v_accvgpr_write_b32 a239, 0
	v_accvgpr_write_b32 a220, 0
	v_accvgpr_write_b32 a221, 0
	v_accvgpr_write_b32 a222, 0
	v_accvgpr_write_b32 a223, 0
	v_accvgpr_write_b32 a12, 0
	v_accvgpr_write_b32 a13, 0
	v_accvgpr_write_b32 a14, 0
	v_accvgpr_write_b32 a15, 0
	v_accvgpr_write_b32 a216, 0
	v_accvgpr_write_b32 a217, 0
	v_accvgpr_write_b32 a218, 0
	v_accvgpr_write_b32 a219, 0
	v_accvgpr_write_b32 a144, 0
	v_accvgpr_write_b32 a145, 0
	v_accvgpr_write_b32 a146, 0
	v_accvgpr_write_b32 a147, 0
	v_accvgpr_write_b32 a148, 0
	v_accvgpr_write_b32 a149, 0
	v_accvgpr_write_b32 a150, 0
	v_accvgpr_write_b32 a151, 0
	v_mov_b32_e32 v144, 0
	v_mov_b32_e32 v145, 0
	v_mov_b32_e32 v146, 0
	v_mov_b32_e32 v147, 0
	v_mov_b32_e32 v118, 0
	v_mov_b32_e32 v119, 0
	v_mov_b32_e32 v120, 0
	v_mov_b32_e32 v121, 0
	v_mov_b32_e32 v110, 0
	v_mov_b32_e32 v111, 0
	v_mov_b32_e32 v112, 0
	v_mov_b32_e32 v113, 0
	v_mov_b32_e32 v106, 0
	v_mov_b32_e32 v107, 0
	v_mov_b32_e32 v108, 0
	v_mov_b32_e32 v109, 0
	v_mov_b32_e32 v102, 0
	v_mov_b32_e32 v103, 0
	v_mov_b32_e32 v104, 0
	v_mov_b32_e32 v105, 0
	v_mov_b32_e32 v98, 0
	v_mov_b32_e32 v99, 0
	v_mov_b32_e32 v100, 0
	v_mov_b32_e32 v101, 0
	v_mov_b32_e32 v94, 0
	v_mov_b32_e32 v95, 0
	v_mov_b32_e32 v96, 0
	v_mov_b32_e32 v97, 0
	v_accvgpr_write_b32 a168, 0
	v_accvgpr_write_b32 a169, 0
	v_accvgpr_write_b32 a170, 0
	v_accvgpr_write_b32 a171, 0
	v_accvgpr_write_b32 a110, 0
	v_accvgpr_write_b32 a111, 0
	v_accvgpr_write_b32 a112, 0
	v_accvgpr_write_b32 a113, 0
	v_mov_b32_e32 v82, 0
	v_mov_b32_e32 v83, 0
	v_mov_b32_e32 v84, 0
	v_mov_b32_e32 v85, 0
	v_mov_b32_e32 v78, 0
	v_mov_b32_e32 v79, 0
	v_mov_b32_e32 v80, 0
	v_mov_b32_e32 v81, 0
	v_accvgpr_write_b32 a164, 0
	v_accvgpr_write_b32 a165, 0
	v_accvgpr_write_b32 a166, 0
	v_accvgpr_write_b32 a167, 0
	v_mov_b32_e32 v70, 0
	v_mov_b32_e32 v71, 0
	v_mov_b32_e32 v72, 0
	v_mov_b32_e32 v73, 0
	v_mov_b32_e32 v66, 0
	v_mov_b32_e32 v67, 0
	v_mov_b32_e32 v68, 0
	v_mov_b32_e32 v69, 0
	v_accvgpr_write_b32 a204, 0
	v_accvgpr_write_b32 a205, 0
	v_accvgpr_write_b32 a206, 0
	v_accvgpr_write_b32 a207, 0
	v_accvgpr_write_b32 a200, 0
	v_accvgpr_write_b32 a201, 0
	v_accvgpr_write_b32 a202, 0
	v_accvgpr_write_b32 a203, 0
	v_accvgpr_write_b32 a172, 0
	v_accvgpr_write_b32 a173, 0
	v_accvgpr_write_b32 a174, 0
	v_accvgpr_write_b32 a175, 0
	v_accvgpr_write_b32 a152, 0
	v_accvgpr_write_b32 a153, 0
	v_accvgpr_write_b32 a154, 0
	v_accvgpr_write_b32 a155, 0
	v_accvgpr_write_b32 a156, 0
	v_accvgpr_write_b32 a157, 0
	v_accvgpr_write_b32 a158, 0
	v_accvgpr_write_b32 a159, 0
	v_mov_b32_e32 v46, 0
	v_mov_b32_e32 v47, 0
	v_mov_b32_e32 v48, 0
	v_mov_b32_e32 v49, 0
	v_accvgpr_write_b32 a114, 0
	v_accvgpr_write_b32 a115, 0
	v_accvgpr_write_b32 a116, 0
	v_accvgpr_write_b32 a117, 0
	v_mov_b32_e32 v38, 0
	v_mov_b32_e32 v39, 0
	v_mov_b32_e32 v40, 0
	v_mov_b32_e32 v41, 0
	v_mov_b32_e32 v34, 0
	v_mov_b32_e32 v35, 0
	v_mov_b32_e32 v36, 0
	v_mov_b32_e32 v37, 0
	v_accvgpr_write_b32 a143, v2
	v_accvgpr_write_b32 a16, 0
	v_accvgpr_write_b32 a17, 0
	v_accvgpr_write_b32 a18, 0
	v_accvgpr_write_b32 a19, 0
	v_mov_b32_e32 v162, 0
	v_mov_b32_e32 v163, 0
	v_mov_b32_e32 v164, 0
	v_mov_b32_e32 v165, 0
	v_accvgpr_write_b32 a4, 0
	v_accvgpr_write_b32 a5, 0
	v_accvgpr_write_b32 a6, 0
	v_accvgpr_write_b32 a7, 0
	v_accvgpr_write_b32 a0, 0
	v_accvgpr_write_b32 a1, 0
	v_accvgpr_write_b32 a2, 0
	v_accvgpr_write_b32 a3, 0
	v_mov_b32_e32 v42, 0
	v_mov_b32_e32 v43, 0
	v_mov_b32_e32 v44, 0
	v_mov_b32_e32 v45, 0
	v_mov_b32_e32 v158, 0
	v_mov_b32_e32 v159, 0
	v_mov_b32_e32 v160, 0
	v_mov_b32_e32 v161, 0
	v_mov_b32_e32 v166, 0
	v_mov_b32_e32 v167, 0
	v_mov_b32_e32 v168, 0
	v_mov_b32_e32 v169, 0
	v_accvgpr_write_b32 a8, 0
	v_accvgpr_write_b32 a9, 0
	v_accvgpr_write_b32 a10, 0
	v_accvgpr_write_b32 a11, 0
	s_mov_b32 s3, 0x27000
	s_mov_b32 s2, 0x7ffffffe
	v_accvgpr_write_b32 a162, v1
	v_add_u32_e32 v254, 0, v7
	v_add_u32_e32 v255, 0, v0
.LBB0_2:                                ; =>This Inner Loop Header: Depth=1
	v_accvgpr_read_b32 v0, a106
	v_cmp_gt_i32_e32 vcc, s10, v0
	v_bfrev_b32_e32 v7, 1
	v_mov_b64_e32 v[128:129], v[20:21]
	v_accvgpr_read_b32 v6, a133
	v_mov_b64_e32 v[126:127], v[18:19]
	v_cndmask_b32_e32 v18, v7, v6, vcc
	v_accvgpr_read_b32 v6, a134
	v_cndmask_b32_e32 v19, v7, v6, vcc
	v_accvgpr_read_b32 v6, a135
	v_cndmask_b32_e32 v20, v7, v6, vcc
	v_accvgpr_read_b32 v6, a136
	v_cndmask_b32_e32 v21, v7, v6, vcc
	v_accvgpr_read_b32 v6, a137
	v_cndmask_b32_e32 v50, v7, v6, vcc
	v_accvgpr_read_b32 v6, a138
	v_cndmask_b32_e32 v54, v7, v6, vcc
	v_accvgpr_read_b32 v6, a139
	v_cndmask_b32_e32 v58, v7, v6, vcc
	v_accvgpr_read_b32 v6, a140
	v_accvgpr_read_b32 v0, a107
	v_accvgpr_read_b32 v1, a108
	v_accvgpr_read_b32 v2, a109
	v_accvgpr_read_b32 v4, a131
	v_cndmask_b32_e32 v62, v7, v6, vcc
	v_accvgpr_read_b32 v6, a141
	s_and_b32 s1, s12, 0xffff
	s_mov_b32 s0, s11
	v_cndmask_b32_e32 v0, v7, v0, vcc
	v_cndmask_b32_e32 v1, v7, v1, vcc
	v_cndmask_b32_e32 v2, v7, v2, vcc
	v_accvgpr_read_b32 v3, a130
	v_cndmask_b32_e32 v4, v7, v4, vcc
	v_accvgpr_read_b32 v5, a132
	v_cndmask_b32_e32 v150, v7, v6, vcc
	v_accvgpr_read_b32 v6, a142
	v_mov_b64_e32 v[88:89], v[12:13]
	v_accvgpr_write_b32 a215, v37
	v_cndmask_b32_e32 v3, v7, v3, vcc
	v_cndmask_b32_e32 v5, v7, v5, vcc
	v_cndmask_b32_e32 v151, v7, v6, vcc
	buffer_load_dwordx4 v[6:9], v0, s[0:3], 0 offen
	v_mov_b64_e32 v[86:87], v[10:11]
	buffer_load_dwordx4 v[10:13], v1, s[0:3], 0 offen
	buffer_load_dwordx4 v[14:17], v2, s[0:3], 0 offen
	buffer_load_dwordx4 v[26:29], v4, s[0:3], 0 offen
	buffer_load_dwordx4 v[30:33], v5, s[0:3], 0 offen
	v_accvgpr_write_b32 a214, v36
	v_accvgpr_write_b32 a213, v35
	v_accvgpr_write_b32 a212, v34
	buffer_load_dwordx4 v[34:37], v18, s[0:3], 0 offen
	v_accvgpr_write_b32 a183, v169
	v_accvgpr_write_b32 a182, v168
	v_accvgpr_write_b32 a181, v167
	v_accvgpr_write_b32 a180, v166
	v_mov_b64_e32 v[168:169], v[84:85]
	v_mov_b64_e32 v[166:167], v[82:83]
	v_mov_b64_e32 v[84:85], v[24:25]
	v_accvgpr_write_b32 a211, v165
	v_accvgpr_write_b32 a227, v41
	v_mov_b64_e32 v[82:83], v[22:23]
	buffer_load_dwordx4 v[22:25], v3, s[0:3], 0 offen
	v_accvgpr_write_b32 a191, v45
	v_accvgpr_write_b32 a210, v164
	v_accvgpr_write_b32 a209, v163
	v_accvgpr_write_b32 a208, v162
	v_accvgpr_write_b32 a226, v40
	v_accvgpr_write_b32 a225, v39
	v_accvgpr_write_b32 a224, v38
	buffer_load_dwordx4 v[38:41], v19, s[0:3], 0 offen
	s_and_b32 s1, s5, 0xffff
	s_mov_b32 s0, s4
	v_mov_b64_e32 v[164:165], v[48:49]
	v_accvgpr_write_b32 a190, v44
	v_accvgpr_write_b32 a189, v43
	v_accvgpr_write_b32 a188, v42
	buffer_load_dwordx4 v[42:45], v20, s[0:3], 0 offen
	v_mov_b64_e32 v[162:163], v[46:47]
	buffer_load_dwordx4 v[46:49], v21, s[0:3], 0 offen
	s_nop 0
	buffer_load_dwordx4 v[50:53], v50, s[0:3], 0 offen
	s_nop 0
	buffer_load_dwordx4 v[54:57], v54, s[0:3], 0 offen
	s_nop 0
	buffer_load_dwordx4 v[58:61], v58, s[0:3], 0 offen
	s_nop 0
	buffer_load_dwordx4 v[62:65], v62, s[0:3], 0 offen
	v_mov_b64_e32 v[148:149], v[146:147]
	v_mov_b64_e32 v[146:147], v[144:145]
	v_mov_b64_e32 v[144:145], v[72:73]
	v_mov_b64_e32 v[142:143], v[70:71]
	v_accvgpr_read_b32 v130, a216
	v_accvgpr_read_b32 v131, a217
	v_accvgpr_read_b32 v132, a218
	v_accvgpr_read_b32 v133, a219
	v_accvgpr_write_b32 a219, v161
	v_accvgpr_mov_b32 a195, a175
	v_accvgpr_write_b32 a218, v160
	v_accvgpr_write_b32 a217, v159
	v_accvgpr_write_b32 a216, v158
	v_accvgpr_mov_b32 a235, a223
	v_mov_b64_e32 v[160:161], v[68:69]
	v_accvgpr_mov_b32 a194, a174
	v_accvgpr_mov_b32 a193, a173
	v_accvgpr_mov_b32 a192, a172
	v_accvgpr_mov_b32 a179, a7
	v_accvgpr_mov_b32 a187, a3
	v_accvgpr_mov_b32 a175, a15
	v_accvgpr_mov_b32 a231, a19
	v_accvgpr_mov_b32 a234, a222
	v_accvgpr_mov_b32 a233, a221
	v_accvgpr_mov_b32 a232, a220
	v_accvgpr_mov_b32 a223, a11
	v_mov_b64_e32 v[158:159], v[66:67]
	v_accvgpr_read_b32 v0, a143
	v_accvgpr_read_b32 v1, a161
	v_accvgpr_read_b32 v2, a162
	v_accvgpr_mov_b32 a178, a6
	v_accvgpr_mov_b32 a177, a5
	v_accvgpr_mov_b32 a176, a4
	v_accvgpr_mov_b32 a186, a2
	v_accvgpr_mov_b32 a185, a1
	v_accvgpr_mov_b32 a184, a0
	v_accvgpr_mov_b32 a174, a14
	v_accvgpr_mov_b32 a173, a13
	v_accvgpr_mov_b32 a172, a12
	v_accvgpr_mov_b32 a230, a18
	v_accvgpr_mov_b32 a229, a17
	v_accvgpr_mov_b32 a228, a16
	v_accvgpr_mov_b32 a222, a10
	s_waitcnt vmcnt(13)
	v_cndmask_b32_e32 v69, 0, v9, vcc
	v_cndmask_b32_e32 v68, 0, v8, vcc
	s_waitcnt vmcnt(12)
	v_cndmask_b32_e32 v73, 0, v13, vcc
	v_cndmask_b32_e32 v72, 0, v12, vcc
	v_cndmask_b32_e32 v71, 0, v11, vcc
	v_cndmask_b32_e32 v70, 0, v10, vcc
	s_waitcnt vmcnt(11)
	v_cndmask_b32_e32 v77, 0, v17, vcc
	v_cndmask_b32_e32 v76, 0, v16, vcc
	v_cndmask_b32_e32 v75, 0, v15, vcc
	v_cndmask_b32_e32 v74, 0, v14, vcc
	s_waitcnt vmcnt(9)
	v_cndmask_b32_e32 v17, 0, v33, vcc
	v_cndmask_b32_e32 v16, 0, v32, vcc
	v_cndmask_b32_e32 v15, 0, v31, vcc
	v_cndmask_b32_e32 v14, 0, v30, vcc
	s_waitcnt vmcnt(8)
	v_cndmask_b32_e32 v13, 0, v37, vcc
	v_cndmask_b32_e32 v12, 0, v36, vcc
	v_cndmask_b32_e32 v11, 0, v35, vcc
	v_cndmask_b32_e32 v10, 0, v34, vcc
	buffer_load_dwordx4 v[30:33], v150, s[0:3], 0 offen
	buffer_load_dwordx4 v[34:37], v151, s[0:3], 0 offen
	v_cndmask_b32_e32 v67, 0, v7, vcc
	v_cndmask_b32_e32 v66, 0, v6, vcc
	s_waitcnt vmcnt(9)
	v_cndmask_b32_e32 v25, 0, v25, vcc
	v_cndmask_b32_e32 v24, 0, v24, vcc
	v_cndmask_b32_e32 v23, 0, v23, vcc
	v_cndmask_b32_e32 v22, 0, v22, vcc
	v_cndmask_b32_e32 v29, 0, v29, vcc
	v_cndmask_b32_e32 v28, 0, v28, vcc
	v_cndmask_b32_e32 v27, 0, v27, vcc
	v_cndmask_b32_e32 v26, 0, v26, vcc
	s_waitcnt vmcnt(8)
	v_cndmask_b32_e32 v9, 0, v41, vcc
	v_cndmask_b32_e32 v8, 0, v40, vcc
	v_cndmask_b32_e32 v7, 0, v39, vcc
	v_cndmask_b32_e32 v6, 0, v38, vcc
	v_accvgpr_mov_b32 a221, a9
	v_accvgpr_mov_b32 a220, a8
	s_waitcnt lgkmcnt(0)
	s_barrier
	ds_write_b128 v0, v[66:69]
	s_waitcnt vmcnt(7)
	v_cndmask_b32_e32 v41, 0, v45, vcc
	v_cndmask_b32_e32 v40, 0, v44, vcc
	v_cndmask_b32_e32 v39, 0, v43, vcc
	v_cndmask_b32_e32 v38, 0, v42, vcc
	s_waitcnt vmcnt(6)
	v_cndmask_b32_e32 v45, 0, v49, vcc
	v_cndmask_b32_e32 v44, 0, v48, vcc
	v_cndmask_b32_e32 v43, 0, v47, vcc
	v_cndmask_b32_e32 v42, 0, v46, vcc
	s_waitcnt vmcnt(5)
	v_cndmask_b32_e32 v49, 0, v53, vcc
	v_cndmask_b32_e32 v48, 0, v52, vcc
	v_cndmask_b32_e32 v47, 0, v51, vcc
	v_cndmask_b32_e32 v46, 0, v50, vcc
	s_waitcnt vmcnt(4)
	v_cndmask_b32_e32 v53, 0, v57, vcc
	v_cndmask_b32_e32 v52, 0, v56, vcc
	v_cndmask_b32_e32 v51, 0, v55, vcc
	v_cndmask_b32_e32 v50, 0, v54, vcc
	s_waitcnt vmcnt(3)
	v_cndmask_b32_e32 v57, 0, v61, vcc
	v_cndmask_b32_e32 v56, 0, v60, vcc
	v_cndmask_b32_e32 v55, 0, v59, vcc
	v_cndmask_b32_e32 v54, 0, v58, vcc
	s_waitcnt vmcnt(2)
	v_cndmask_b32_e32 v61, 0, v65, vcc
	v_cndmask_b32_e32 v60, 0, v64, vcc
	v_cndmask_b32_e32 v59, 0, v63, vcc
	v_cndmask_b32_e32 v58, 0, v62, vcc
	s_waitcnt lgkmcnt(0)
	s_barrier
	ds_read_b128 v[62:65], v1
	ds_read_b128 a[16:19], v2
	s_waitcnt lgkmcnt(0)
	s_barrier
	ds_write_b128 v0, v[70:73]
	s_waitcnt lgkmcnt(0)
	s_barrier
	ds_read_b128 v[66:69], v1
	ds_read_b128 a[12:15], v2
	s_waitcnt lgkmcnt(0)
	s_barrier
	ds_write_b128 v0, v[74:77]
	s_waitcnt lgkmcnt(0)
	s_barrier
	ds_read_b128 v[70:73], v1
	ds_read_b128 a[8:11], v2
	s_waitcnt lgkmcnt(0)
	s_barrier
	ds_write_b128 v0, v[22:25]
	s_waitcnt lgkmcnt(0)
	s_barrier
	ds_read_b128 v[74:77], v1
	ds_read_b128 a[4:7], v2
	s_waitcnt lgkmcnt(0)
	s_barrier
	ds_write_b128 v0, v[26:29]
	s_waitcnt lgkmcnt(0)
	s_barrier
	ds_read_b128 a[56:59], v1
	ds_read_b128 a[0:3], v2
	s_waitcnt lgkmcnt(0)
	s_barrier
	ds_write_b128 v0, v[14:17]
	s_waitcnt lgkmcnt(0)
	s_barrier
	ds_read_b128 a[60:63], v1
	ds_read_b128 v[14:17], v2
	s_waitcnt lgkmcnt(0)
	s_barrier
	ds_write_b128 v0, v[10:13]
	s_waitcnt lgkmcnt(0)
	s_barrier
	ds_read_b128 a[64:67], v1
	ds_read_b128 a[52:55], v2
	s_waitcnt lgkmcnt(0)
	s_barrier
	ds_write_b128 v0, v[6:9]
	v_accvgpr_read_b32 v0, a160
	s_waitcnt lgkmcnt(0)
	s_barrier
	ds_read_b128 a[68:71], v1
	ds_read_b128 a[48:51], v2
	s_waitcnt lgkmcnt(0)
	s_barrier
	ds_write_b128 v0, v[38:41]
	s_waitcnt lgkmcnt(0)
	s_barrier
	ds_read_b128 a[72:75], v254
	ds_read_b128 a[20:23], v255 offset:512
	v_accvgpr_write_b32 a199, v81
	v_accvgpr_write_b32 a198, v80
	v_accvgpr_write_b32 a197, v79
	v_accvgpr_write_b32 a196, v78
	v_accvgpr_read_b32 v78, a100
	v_accvgpr_read_b32 v79, a101
	v_accvgpr_read_b32 v80, a102
	v_accvgpr_read_b32 v81, a103
	s_waitcnt vmcnt(1)
	v_cndmask_b32_e32 v33, 0, v33, vcc
	v_cndmask_b32_e32 v32, 0, v32, vcc
	v_cndmask_b32_e32 v31, 0, v31, vcc
	v_cndmask_b32_e32 v30, 0, v30, vcc
	s_waitcnt lgkmcnt(0)
	s_barrier
	ds_write_b128 v0, v[42:45]
	s_waitcnt lgkmcnt(0)
	s_barrier
	ds_read_b128 a[76:79], v254
	ds_read_b128 a[24:27], v255 offset:512
	s_waitcnt lgkmcnt(0)
	s_barrier
	ds_write_b128 v0, v[46:49]
	s_waitcnt lgkmcnt(0)
	s_barrier
	ds_read_b128 a[80:83], v254
	ds_read_b128 a[28:31], v255 offset:512
	s_waitcnt lgkmcnt(0)
	s_barrier
	ds_write_b128 v0, v[50:53]
	s_waitcnt lgkmcnt(0)
	s_barrier
	ds_read_b128 a[84:87], v254
	ds_read_b128 a[32:35], v255 offset:512
	s_waitcnt lgkmcnt(0)
	s_barrier
	ds_write_b128 v0, v[54:57]
	s_waitcnt lgkmcnt(0)
	s_barrier
	ds_read_b128 a[88:91], v254
	ds_read_b128 a[36:39], v255 offset:512
	s_waitcnt lgkmcnt(0)
	s_barrier
	ds_write_b128 v0, v[58:61]
	s_waitcnt lgkmcnt(0)
	s_barrier
	ds_read_b128 a[92:95], v254
	ds_read_b128 a[40:43], v255 offset:512
	s_waitcnt lgkmcnt(0)
	s_barrier
	ds_write_b128 v0, v[30:33]
	v_mfma_f32_16x16x32_f16 v[30:33], a[72:75], v[66:69], v[78:81]
	v_accvgpr_read_b32 v90, a96
	v_accvgpr_read_b32 v125, a113
	v_accvgpr_read_b32 v91, a97
	v_accvgpr_read_b32 v92, a98
	v_accvgpr_read_b32 v93, a99
	v_accvgpr_read_b32 v124, a112
	v_accvgpr_read_b32 v123, a111
	v_accvgpr_read_b32 v122, a110
	v_accvgpr_write_b32 a113, v33
	v_accvgpr_write_b32 a112, v32
	v_accvgpr_write_b32 a111, v31
	v_accvgpr_write_b32 a110, v30
	v_mfma_f32_16x16x32_f16 v[30:33], a[76:79], v[66:69], v[90:93]
	s_waitcnt lgkmcnt(0)
	s_barrier
	ds_read_b128 a[96:99], v254
	ds_read_b128 a[44:47], v255 offset:512
	v_accvgpr_read_b32 v177, a117
	v_accvgpr_read_b32 v176, a116
	v_accvgpr_read_b32 v175, a115
	v_accvgpr_read_b32 v174, a114
	s_waitcnt vmcnt(0)
	v_cndmask_b32_e32 v37, 0, v37, vcc
	v_cndmask_b32_e32 v36, 0, v36, vcc
	v_cndmask_b32_e32 v35, 0, v35, vcc
	v_cndmask_b32_e32 v34, 0, v34, vcc
	v_accvgpr_write_b32 a117, v33
	s_waitcnt lgkmcnt(0)
	s_barrier
	ds_write_b128 v0, v[34:37]
	s_waitcnt lgkmcnt(0)
	s_barrier
	ds_read_b128 a[100:103], v254
	v_accvgpr_write_b32 a116, v32
	v_accvgpr_write_b32 a115, v31
	v_accvgpr_write_b32 a114, v30
	v_mfma_f32_16x16x32_f16 v[30:33], a[96:99], v[74:77], v[130:133]
	v_accvgpr_read_b32 v114, a168
	v_accvgpr_read_b32 v138, a164
	v_accvgpr_read_b32 v115, a169
	v_accvgpr_read_b32 v116, a170
	v_accvgpr_read_b32 v117, a171
	v_accvgpr_read_b32 v139, a165
	v_accvgpr_read_b32 v140, a166
	v_accvgpr_read_b32 v141, a167
	v_accvgpr_write_b32 a167, v33
	v_mfma_f32_16x16x32_f16 v[22:25], a[76:79], v[62:65], v[154:157]
	v_accvgpr_write_b32 a166, v32
	v_accvgpr_write_b32 a165, v31
	v_accvgpr_write_b32 a164, v30
	v_mfma_f32_16x16x32_f16 v[0:3], a[80:83], v[62:65], v[82:85]
	v_accvgpr_read_b32 v90, a208
	v_accvgpr_read_b32 v91, a209
	v_accvgpr_read_b32 v92, a210
	v_mfma_f32_16x16x32_f16 v[30:33], a[72:75], a[60:63], v[114:117]
	v_accvgpr_read_b32 v93, a211
	s_add_u32 s11, s11, 0x80
	s_addc_u32 s12, s12, 0
	v_mfma_f32_16x16x32_f16 a[148:151], a[72:75], a[56:59], a[148:151]
	s_add_u32 s4, s4, 0x80
	s_addc_u32 s5, s5, 0
	s_add_i32 s17, s17, -1
	v_mfma_f32_16x16x32_f16 v[146:149], a[76:79], a[56:59], v[146:149]
	s_sub_i32 s10, s10, 64
	s_cmp_lg_u32 s17, 0
	v_mfma_f32_16x16x32_f16 v[118:121], a[80:83], a[56:59], v[118:121]
	v_mfma_f32_16x16x32_f16 v[110:113], a[84:87], a[56:59], v[110:113]
	v_mfma_f32_16x16x32_f16 v[106:109], a[88:91], a[56:59], v[106:109]
	v_mfma_f32_16x16x32_f16 v[102:105], a[92:95], a[56:59], v[102:105]
	v_mfma_f32_16x16x32_f16 v[98:101], a[96:99], a[56:59], v[98:101]
	v_mfma_f32_16x16x32_f16 v[154:157], a[24:27], a[16:19], v[22:25]
	v_mfma_f32_16x16x32_f16 v[22:25], a[28:31], a[16:19], v[0:3]
	s_nop 2
	ds_read_b128 v[0:3], v255 offset:512
	s_waitcnt lgkmcnt(1)
	v_mfma_f32_16x16x32_f16 v[94:97], a[100:103], a[56:59], v[94:97]
	v_accvgpr_write_b32 a59, v33
	v_accvgpr_write_b32 a58, v32
	v_accvgpr_write_b32 a57, v31
	v_accvgpr_write_b32 a56, v30
	v_mfma_f32_16x16x32_f16 v[30:33], a[76:79], a[60:63], v[122:125]
	v_mfma_f32_16x16x32_f16 a[168:171], a[92:95], v[74:77], a[172:175]
	v_mfma_f32_16x16x32_f16 v[4:7], a[72:75], v[62:65], v[134:137]
	s_nop 5
	v_accvgpr_write_b32 a175, v33
	v_accvgpr_write_b32 a174, v32
	v_accvgpr_write_b32 a173, v31
	v_accvgpr_write_b32 a172, v30
	v_accvgpr_read_b32 v30, a196
	v_mfma_f32_16x16x32_f16 a[248:251], a[72:75], v[74:77], a[248:251]
	v_accvgpr_read_b32 v31, a197
	v_accvgpr_read_b32 v32, a198
	v_accvgpr_read_b32 v33, a199
	v_mfma_f32_16x16x32_f16 a[244:247], a[76:79], v[74:77], a[244:247]
	v_mfma_f32_16x16x32_f16 a[240:243], a[80:83], v[74:77], a[240:243]
	v_mfma_f32_16x16x32_f16 a[236:239], a[84:87], v[74:77], a[236:239]
	v_mfma_f32_16x16x32_f16 a[232:235], a[88:91], v[74:77], a[232:235]
	v_mfma_f32_16x16x32_f16 a[144:147], a[100:103], v[74:77], a[144:147]
	v_mfma_f32_16x16x32_f16 v[74:77], a[88:91], a[60:63], v[138:141]
	v_mfma_f32_16x16x32_f16 v[250:253], a[80:83], v[66:69], v[250:253]
	v_mfma_f32_16x16x32_f16 v[246:249], a[84:87], v[66:69], v[246:249]
	v_mfma_f32_16x16x32_f16 v[242:245], a[88:91], v[66:69], v[242:245]
	v_mfma_f32_16x16x32_f16 v[238:241], a[92:95], v[66:69], v[238:241]
	v_mfma_f32_16x16x32_f16 v[234:237], a[96:99], v[66:69], v[234:237]
	v_mfma_f32_16x16x32_f16 v[230:233], a[100:103], v[66:69], v[230:233]
	v_mfma_f32_16x16x32_f16 v[226:229], a[72:75], v[70:73], v[226:229]
	v_mfma_f32_16x16x32_f16 v[222:225], a[76:79], v[70:73], v[222:225]
	v_mfma_f32_16x16x32_f16 v[218:221], a[80:83], v[70:73], v[218:221]
	v_mfma_f32_16x16x32_f16 v[214:217], a[84:87], v[70:73], v[214:217]
	v_mfma_f32_16x16x32_f16 v[210:213], a[88:91], v[70:73], v[210:213]
	v_mfma_f32_16x16x32_f16 v[206:209], a[92:95], v[70:73], v[206:209]
	v_mfma_f32_16x16x32_f16 v[202:205], a[96:99], v[70:73], v[202:205]
	v_mfma_f32_16x16x32_f16 a[252:255], a[100:103], v[70:73], a[252:255]
	v_mfma_f32_16x16x32_f16 v[82:85], a[80:83], a[60:63], v[166:169]
	v_mfma_f32_16x16x32_f16 v[78:81], a[84:87], a[60:63], v[30:33]
	v_mfma_f32_16x16x32_f16 v[70:73], a[92:95], a[60:63], v[142:145]
	s_nop 1
	v_accvgpr_read_b32 v30, a224
	v_accvgpr_read_b32 v31, a225
	v_accvgpr_read_b32 v32, a226
	v_mfma_f32_16x16x32_f16 v[66:69], a[96:99], a[60:63], v[158:161]
	v_accvgpr_read_b32 v33, a227
	v_mfma_f32_16x16x32_f16 a[204:207], a[100:103], a[60:63], a[204:207]
	v_accvgpr_mov_b32 a60, a228
	v_accvgpr_mov_b32 a61, a229
	v_accvgpr_mov_b32 a62, a230
	v_mfma_f32_16x16x32_f16 v[158:161], a[76:79], a[68:71], v[90:93]
	v_accvgpr_mov_b32 a63, a231
	s_nop 1
	v_accvgpr_read_b32 v90, a188
	v_accvgpr_read_b32 v91, a189
	v_accvgpr_read_b32 v92, a190
	v_accvgpr_read_b32 v93, a191
	v_mfma_f32_16x16x32_f16 a[196:199], a[76:79], a[64:67], a[192:195]
	v_mfma_f32_16x16x32_f16 v[34:37], a[96:99], a[64:67], v[30:33]
	s_nop 2
	v_accvgpr_read_b32 v30, a212
	v_mfma_f32_16x16x32_f16 a[192:195], a[72:75], a[68:71], a[60:63]
	v_accvgpr_read_b32 v31, a213
	v_accvgpr_read_b32 v32, a214
	v_accvgpr_read_b32 v33, a215
	v_accvgpr_mov_b32 a60, a176
	v_mfma_f32_16x16x32_f16 v[130:133], a[88:91], a[68:71], v[90:93]
	v_accvgpr_mov_b32 a61, a177
	v_accvgpr_mov_b32 a62, a178
	v_accvgpr_mov_b32 a63, a179
	v_accvgpr_read_b32 v90, a216
	v_accvgpr_read_b32 v91, a217
	v_accvgpr_read_b32 v92, a218
	v_accvgpr_read_b32 v93, a219
	v_mfma_f32_16x16x32_f16 v[134:137], a[20:23], a[16:19], v[4:7]
	v_mfma_f32_16x16x32_f16 v[4:7], a[36:39], v[14:17], v[74:77]
	v_mfma_f32_16x16x32_f16 a[200:203], a[72:75], a[64:67], a[200:203]
	v_accvgpr_mov_b32 a72, a220
	v_accvgpr_mov_b32 a73, a221
	v_accvgpr_mov_b32 a74, a222
	v_mfma_f32_16x16x32_f16 a[152:155], a[80:83], a[64:67], a[152:155]
	v_accvgpr_mov_b32 a75, a223
	v_mfma_f32_16x16x32_f16 a[156:159], a[84:87], a[64:67], a[156:159]
	v_mfma_f32_16x16x32_f16 v[42:45], a[88:91], a[64:67], v[162:165]
	v_mfma_f32_16x16x32_f16 v[38:41], a[92:95], a[64:67], v[174:177]
	v_mfma_f32_16x16x32_f16 v[30:33], a[100:103], a[64:67], v[30:33]
	v_mfma_f32_16x16x32_f16 a[64:67], a[80:83], a[68:71], a[60:63]
	s_nop 2
	v_accvgpr_mov_b32 a60, a184
	v_mfma_f32_16x16x32_f16 v[122:125], a[92:95], a[68:71], v[90:93]
	v_accvgpr_mov_b32 a61, a185
	v_accvgpr_mov_b32 a62, a186
	v_accvgpr_mov_b32 a63, a187
	v_accvgpr_read_b32 v90, a180
	v_accvgpr_read_b32 v91, a181
	v_accvgpr_read_b32 v92, a182
	v_accvgpr_read_b32 v93, a183
	v_mfma_f32_16x16x32_f16 a[126:129], a[92:95], v[62:65], a[126:129]
	v_mfma_f32_16x16x32_f16 a[122:125], a[88:91], v[62:65], a[122:125]
	v_mfma_f32_16x16x32_f16 a[118:121], a[84:87], v[62:65], a[118:121]
	v_mfma_f32_16x16x32_f16 v[8:11], a[96:99], v[62:65], v[126:129]
	v_mfma_f32_16x16x32_f16 v[86:89], a[100:103], v[62:65], v[86:89]
	v_mfma_f32_16x16x32_f16 a[60:63], a[84:87], a[68:71], a[60:63]
	v_mfma_f32_16x16x32_f16 v[114:117], a[96:99], a[68:71], v[90:93]
	v_accvgpr_mov_b32 a96, a114
	v_accvgpr_mov_b32 a97, a115
	v_accvgpr_mov_b32 a98, a116
	v_mfma_f32_16x16x32_f16 a[68:71], a[100:103], a[68:71], a[72:75]
	v_accvgpr_mov_b32 a100, a110
	v_accvgpr_mov_b32 a101, a111
	v_accvgpr_mov_b32 a102, a112
	v_mfma_f32_16x16x32_f16 v[226:229], a[20:23], a[8:11], v[226:229]
	v_accvgpr_mov_b32 a103, a113
	v_accvgpr_mov_b32 a99, a117
	v_accvgpr_mov_b32 a110, a172
	v_mfma_f32_16x16x32_f16 v[222:225], a[24:27], a[8:11], v[222:225]
	v_accvgpr_mov_b32 a111, a173
	v_accvgpr_mov_b32 a112, a174
	v_accvgpr_mov_b32 a113, a175
	v_mfma_f32_16x16x32_f16 v[218:221], a[28:31], a[8:11], v[218:221]
	v_mfma_f32_16x16x32_f16 v[214:217], a[32:35], a[8:11], v[214:217]
	v_mfma_f32_16x16x32_f16 v[210:213], a[36:39], a[8:11], v[210:213]
	v_mfma_f32_16x16x32_f16 v[206:209], a[40:43], a[8:11], v[206:209]
	v_mfma_f32_16x16x32_f16 v[202:205], a[44:47], a[8:11], v[202:205]
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_16x16x32_f16 a[252:255], v[0:3], a[8:11], a[252:255]
	v_accvgpr_mov_b32 a8, a164
	v_accvgpr_mov_b32 a9, a165
	v_accvgpr_mov_b32 a10, a166
	v_accvgpr_mov_b32 a11, a167
	v_accvgpr_write_b32 a167, v7
	v_mfma_f32_16x16x32_f16 a[100:103], a[20:23], a[12:15], a[100:103]
	v_accvgpr_write_b32 a166, v6
	v_accvgpr_write_b32 a165, v5
	v_accvgpr_write_b32 a164, v4
	v_mfma_f32_16x16x32_f16 a[96:99], a[24:27], a[12:15], a[96:99]
	v_mfma_f32_16x16x32_f16 v[250:253], a[28:31], a[12:15], v[250:253]
	v_mfma_f32_16x16x32_f16 v[246:249], a[32:35], a[12:15], v[246:249]
	v_mfma_f32_16x16x32_f16 v[242:245], a[36:39], a[12:15], v[242:245]
	v_mfma_f32_16x16x32_f16 v[238:241], a[40:43], a[12:15], v[238:241]
	v_mfma_f32_16x16x32_f16 v[234:237], a[44:47], a[12:15], v[234:237]
	v_mfma_f32_16x16x32_f16 v[230:233], v[0:3], a[12:15], v[230:233]
	v_mfma_f32_16x16x32_f16 a[12:15], a[40:43], a[4:7], a[168:171]
	s_nop 2
	v_accvgpr_mov_b32 a171, a59
	v_mfma_f32_16x16x32_f16 v[4:7], a[40:43], a[52:55], v[38:41]
	v_accvgpr_mov_b32 a170, a58
	v_accvgpr_mov_b32 a169, a57
	v_accvgpr_mov_b32 a168, a56
	v_mfma_f32_16x16x32_f16 a[118:121], a[32:35], a[16:19], a[118:121]
	v_mfma_f32_16x16x32_f16 a[122:125], a[36:39], a[16:19], a[122:125]
	s_nop 2
	v_accvgpr_write_b32 a117, v7
	v_accvgpr_write_b32 a116, v6
	v_accvgpr_write_b32 a115, v5
	v_mfma_f32_16x16x32_f16 a[126:129], a[40:43], a[16:19], a[126:129]
	v_accvgpr_write_b32 a114, v4
	v_mfma_f32_16x16x32_f16 v[18:21], a[44:47], a[16:19], v[8:11]
	v_mfma_f32_16x16x32_f16 v[10:13], v[0:3], a[16:19], v[86:89]
	v_mfma_f32_16x16x32_f16 a[248:251], a[20:23], a[4:7], a[248:251]
	v_mfma_f32_16x16x32_f16 a[244:247], a[24:27], a[4:7], a[244:247]
	v_mfma_f32_16x16x32_f16 a[240:243], a[28:31], a[4:7], a[240:243]
	v_mfma_f32_16x16x32_f16 a[236:239], a[32:35], a[4:7], a[236:239]
	v_mfma_f32_16x16x32_f16 a[220:223], a[36:39], a[4:7], a[232:235]
	v_mfma_f32_16x16x32_f16 a[216:219], a[44:47], a[4:7], a[8:11]
	v_mfma_f32_16x16x32_f16 a[144:147], v[0:3], a[4:7], a[144:147]
	v_mfma_f32_16x16x32_f16 a[148:151], a[20:23], a[0:3], a[148:151]
	v_mfma_f32_16x16x32_f16 v[144:147], a[24:27], a[0:3], v[146:149]
	v_mfma_f32_16x16x32_f16 v[118:121], a[28:31], a[0:3], v[118:121]
	v_mfma_f32_16x16x32_f16 v[110:113], a[32:35], a[0:3], v[110:113]
	v_mfma_f32_16x16x32_f16 v[106:109], a[36:39], a[0:3], v[106:109]
	v_mfma_f32_16x16x32_f16 v[102:105], a[40:43], a[0:3], v[102:105]
	v_mfma_f32_16x16x32_f16 v[98:101], a[44:47], a[0:3], v[98:101]
	v_mfma_f32_16x16x32_f16 v[94:97], v[0:3], a[0:3], v[94:97]
	v_mfma_f32_16x16x32_f16 a[168:171], a[20:23], v[14:17], a[168:171]
	v_mfma_f32_16x16x32_f16 a[110:113], a[24:27], v[14:17], a[110:113]
	v_mfma_f32_16x16x32_f16 v[82:85], a[28:31], v[14:17], v[82:85]
	v_mfma_f32_16x16x32_f16 v[78:81], a[32:35], v[14:17], v[78:81]
	v_mfma_f32_16x16x32_f16 v[70:73], a[40:43], v[14:17], v[70:73]
	v_mfma_f32_16x16x32_f16 v[66:69], a[44:47], v[14:17], v[66:69]
	v_mfma_f32_16x16x32_f16 a[204:207], v[0:3], v[14:17], a[204:207]
	v_mfma_f32_16x16x32_f16 a[200:203], a[20:23], a[52:55], a[200:203]
	v_mfma_f32_16x16x32_f16 a[172:175], a[24:27], a[52:55], a[196:199]
	v_mfma_f32_16x16x32_f16 a[152:155], a[28:31], a[52:55], a[152:155]
	v_mfma_f32_16x16x32_f16 a[156:159], a[32:35], a[52:55], a[156:159]
	v_mfma_f32_16x16x32_f16 v[46:49], a[36:39], a[52:55], v[42:45]
	v_mfma_f32_16x16x32_f16 v[38:41], a[44:47], a[52:55], v[34:37]
	v_mfma_f32_16x16x32_f16 v[34:37], v[0:3], a[52:55], v[30:33]
	v_mfma_f32_16x16x32_f16 a[16:19], a[20:23], a[48:51], a[192:195]
	v_mfma_f32_16x16x32_f16 v[162:165], a[24:27], a[48:51], v[158:161]
	v_mfma_f32_16x16x32_f16 a[4:7], a[28:31], a[48:51], a[64:67]
	v_mfma_f32_16x16x32_f16 a[0:3], a[32:35], a[48:51], a[60:63]
	v_mfma_f32_16x16x32_f16 v[42:45], a[36:39], a[48:51], v[130:133]
	v_mfma_f32_16x16x32_f16 v[158:161], a[40:43], a[48:51], v[122:125]
	v_mfma_f32_16x16x32_f16 v[166:169], a[44:47], a[48:51], v[114:117]
	v_mfma_f32_16x16x32_f16 a[8:11], v[0:3], a[48:51], a[68:71]
	s_cbranch_scc1 .LBB0_2
; %bb.3:                                ; %Flow
	v_mov_b64_e32 v[2:3], v[134:135]
	v_accvgpr_write_b32 a109, v13
	v_mov_b64_e32 v[4:5], v[136:137]
	v_accvgpr_read_b32 v6, a118
	v_accvgpr_read_b32 v26, a122
	v_accvgpr_read_b32 v30, a126
	v_accvgpr_read_b32 v114, a156
	v_accvgpr_read_b32 v122, a152
	v_accvgpr_read_b32 v136, a148
	v_accvgpr_read_b32 v132, a144
	v_accvgpr_write_b32 a108, v12
	v_accvgpr_write_b32 a107, v11
	v_accvgpr_write_b32 a106, v10
	v_accvgpr_read_b32 v0, a104
	v_accvgpr_read_b32 v50, a105
	v_accvgpr_read_b32 v7, a119
	v_accvgpr_read_b32 v8, a120
	v_accvgpr_read_b32 v9, a121
	v_accvgpr_read_b32 v27, a123
	v_accvgpr_read_b32 v28, a124
	v_accvgpr_read_b32 v29, a125
	v_accvgpr_read_b32 v31, a127
	v_accvgpr_read_b32 v32, a128
	v_accvgpr_read_b32 v33, a129
	v_accvgpr_read_b32 v115, a157
	v_accvgpr_read_b32 v116, a158
	v_accvgpr_read_b32 v117, a159
	v_accvgpr_read_b32 v123, a153
	v_accvgpr_read_b32 v124, a154
	v_accvgpr_read_b32 v125, a155
	v_accvgpr_read_b32 v137, a149
	v_accvgpr_read_b32 v138, a150
	v_accvgpr_read_b32 v139, a151
	v_accvgpr_read_b32 v133, a145
	v_accvgpr_read_b32 v134, a146
	v_accvgpr_read_b32 v135, a147
	s_branch .LBB0_5
.LBB0_4:
	v_accvgpr_write_b32 a11, 0
	v_accvgpr_write_b32 a10, 0
	v_accvgpr_write_b32 a9, 0
	v_accvgpr_write_b32 a8, 0
	v_mov_b32_e32 v169, 0
	v_mov_b32_e32 v168, 0
	v_mov_b32_e32 v167, 0
	v_mov_b32_e32 v166, 0
	v_mov_b32_e32 v161, 0
	v_mov_b32_e32 v160, 0
	v_mov_b32_e32 v159, 0
	v_mov_b32_e32 v158, 0
	v_mov_b32_e32 v45, 0
	v_mov_b32_e32 v44, 0
	v_mov_b32_e32 v43, 0
	v_mov_b32_e32 v42, 0
	v_accvgpr_write_b32 a3, 0
	v_accvgpr_write_b32 a2, 0
	v_accvgpr_write_b32 a1, 0
	v_accvgpr_write_b32 a0, 0
	v_accvgpr_write_b32 a7, 0
	v_accvgpr_write_b32 a6, 0
	v_accvgpr_write_b32 a5, 0
	v_accvgpr_write_b32 a4, 0
	v_mov_b32_e32 v165, 0
	v_mov_b32_e32 v164, 0
	v_mov_b32_e32 v163, 0
	v_mov_b32_e32 v162, 0
	v_accvgpr_write_b32 a19, 0
	v_accvgpr_write_b32 a18, 0
	v_accvgpr_write_b32 a17, 0
	v_accvgpr_write_b32 a16, 0
	v_mov_b32_e32 v37, 0
	v_mov_b32_e32 v36, 0
	v_mov_b32_e32 v35, 0
	v_mov_b32_e32 v34, 0
	v_mov_b32_e32 v41, 0
	v_mov_b32_e32 v40, 0
	v_mov_b32_e32 v39, 0
	v_mov_b32_e32 v38, 0
	v_accvgpr_write_b32 a117, 0
	v_accvgpr_write_b32 a116, 0
	v_accvgpr_write_b32 a115, 0
	v_accvgpr_write_b32 a114, 0
	v_mov_b32_e32 v49, 0
	v_mov_b32_e32 v48, 0
	v_mov_b32_e32 v47, 0
	v_mov_b32_e32 v46, 0
	v_mov_b32_e32 v117, 0
	v_mov_b32_e32 v116, 0
	v_mov_b32_e32 v115, 0
	v_mov_b32_e32 v114, 0
	v_mov_b32_e32 v125, 0
	v_mov_b32_e32 v124, 0
	v_mov_b32_e32 v123, 0
	v_mov_b32_e32 v122, 0
	v_accvgpr_write_b32 a175, 0
	v_accvgpr_write_b32 a174, 0
	v_accvgpr_write_b32 a173, 0
	v_accvgpr_write_b32 a172, 0
	v_accvgpr_write_b32 a203, 0
	v_accvgpr_write_b32 a202, 0
	v_accvgpr_write_b32 a201, 0
	v_accvgpr_write_b32 a200, 0
	v_accvgpr_write_b32 a207, 0
	v_accvgpr_write_b32 a206, 0
	v_accvgpr_write_b32 a205, 0
	v_accvgpr_write_b32 a204, 0
	v_mov_b32_e32 v69, 0
	v_mov_b32_e32 v68, 0
	v_mov_b32_e32 v67, 0
	v_mov_b32_e32 v66, 0
	v_mov_b32_e32 v73, 0
	v_mov_b32_e32 v72, 0
	v_mov_b32_e32 v71, 0
	v_mov_b32_e32 v70, 0
	v_accvgpr_write_b32 a167, 0
	v_accvgpr_write_b32 a166, 0
	v_accvgpr_write_b32 a165, 0
	v_accvgpr_write_b32 a164, 0
	v_mov_b32_e32 v81, 0
	v_mov_b32_e32 v80, 0
	v_mov_b32_e32 v79, 0
	v_mov_b32_e32 v78, 0
	v_mov_b32_e32 v85, 0
	v_mov_b32_e32 v84, 0
	v_mov_b32_e32 v83, 0
	v_mov_b32_e32 v82, 0
	v_accvgpr_write_b32 a113, 0
	v_accvgpr_write_b32 a112, 0
	v_accvgpr_write_b32 a111, 0
	v_accvgpr_write_b32 a110, 0
	v_accvgpr_write_b32 a171, 0
	v_accvgpr_write_b32 a170, 0
	v_accvgpr_write_b32 a169, 0
	v_accvgpr_write_b32 a168, 0
	v_mov_b32_e32 v97, 0
	v_mov_b32_e32 v96, 0
	v_mov_b32_e32 v95, 0
	v_mov_b32_e32 v94, 0
	v_mov_b32_e32 v101, 0
	v_mov_b32_e32 v100, 0
	v_mov_b32_e32 v99, 0
	v_mov_b32_e32 v98, 0
	v_mov_b32_e32 v105, 0
	v_mov_b32_e32 v104, 0
	v_mov_b32_e32 v103, 0
	v_mov_b32_e32 v102, 0
	v_mov_b32_e32 v109, 0
	v_mov_b32_e32 v108, 0
	v_mov_b32_e32 v107, 0
	v_mov_b32_e32 v106, 0
	v_mov_b32_e32 v113, 0
	v_mov_b32_e32 v112, 0
	v_mov_b32_e32 v111, 0
	v_mov_b32_e32 v110, 0
	v_mov_b32_e32 v121, 0
	v_mov_b32_e32 v120, 0
	v_mov_b32_e32 v119, 0
	v_mov_b32_e32 v118, 0
	v_mov_b32_e32 v147, 0
	v_mov_b32_e32 v146, 0
	v_mov_b32_e32 v145, 0
	v_mov_b32_e32 v144, 0
	v_mov_b32_e32 v139, 0
	v_mov_b32_e32 v138, 0
	v_mov_b32_e32 v137, 0
	v_mov_b32_e32 v136, 0
	v_mov_b32_e32 v135, 0
	v_mov_b32_e32 v134, 0
	v_mov_b32_e32 v133, 0
	v_mov_b32_e32 v132, 0
	v_accvgpr_write_b32 a219, 0
	v_accvgpr_write_b32 a218, 0
	v_accvgpr_write_b32 a217, 0
	v_accvgpr_write_b32 a216, 0
	v_accvgpr_write_b32 a15, 0
	v_accvgpr_write_b32 a14, 0
	v_accvgpr_write_b32 a13, 0
	v_accvgpr_write_b32 a12, 0
	v_accvgpr_write_b32 a223, 0
	v_accvgpr_write_b32 a222, 0
	v_accvgpr_write_b32 a221, 0
	v_accvgpr_write_b32 a220, 0
	v_accvgpr_write_b32 a239, 0
	v_accvgpr_write_b32 a238, 0
	v_accvgpr_write_b32 a237, 0
	v_accvgpr_write_b32 a236, 0
	v_accvgpr_write_b32 a243, 0
	v_accvgpr_write_b32 a242, 0
	v_accvgpr_write_b32 a241, 0
	v_accvgpr_write_b32 a240, 0
	v_accvgpr_write_b32 a247, 0
	v_accvgpr_write_b32 a246, 0
	v_accvgpr_write_b32 a245, 0
	v_accvgpr_write_b32 a244, 0
	v_accvgpr_write_b32 a251, 0
	v_accvgpr_write_b32 a250, 0
	v_accvgpr_write_b32 a249, 0
	v_accvgpr_write_b32 a248, 0
	v_accvgpr_write_b32 a255, 0
	v_accvgpr_write_b32 a254, 0
	v_accvgpr_write_b32 a253, 0
	v_accvgpr_write_b32 a252, 0
	v_mov_b32_e32 v205, 0
	v_mov_b32_e32 v204, 0
	v_mov_b32_e32 v203, 0
	v_mov_b32_e32 v202, 0
	v_mov_b32_e32 v209, 0
	v_mov_b32_e32 v208, 0
	v_mov_b32_e32 v207, 0
	v_mov_b32_e32 v206, 0
	v_mov_b32_e32 v213, 0
	v_mov_b32_e32 v212, 0
	v_mov_b32_e32 v211, 0
	v_mov_b32_e32 v210, 0
	v_mov_b32_e32 v217, 0
	v_mov_b32_e32 v216, 0
	v_mov_b32_e32 v215, 0
	v_mov_b32_e32 v214, 0
	v_mov_b32_e32 v221, 0
	v_mov_b32_e32 v220, 0
	v_mov_b32_e32 v219, 0
	v_mov_b32_e32 v218, 0
	v_mov_b32_e32 v225, 0
	v_mov_b32_e32 v224, 0
	v_mov_b32_e32 v223, 0
	v_mov_b32_e32 v222, 0
	v_mov_b32_e32 v229, 0
	v_mov_b32_e32 v228, 0
	v_mov_b32_e32 v227, 0
	v_mov_b32_e32 v226, 0
	v_mov_b32_e32 v233, 0
	v_mov_b32_e32 v232, 0
	v_mov_b32_e32 v231, 0
	v_mov_b32_e32 v230, 0
	v_mov_b32_e32 v237, 0
	v_mov_b32_e32 v236, 0
	v_mov_b32_e32 v235, 0
	v_mov_b32_e32 v234, 0
	v_mov_b32_e32 v241, 0
	v_mov_b32_e32 v240, 0
	v_mov_b32_e32 v239, 0
	v_mov_b32_e32 v238, 0
	v_mov_b32_e32 v245, 0
	v_mov_b32_e32 v244, 0
	v_mov_b32_e32 v243, 0
	v_mov_b32_e32 v242, 0
	v_mov_b32_e32 v249, 0
	v_mov_b32_e32 v248, 0
	v_mov_b32_e32 v247, 0
	v_mov_b32_e32 v246, 0
	v_mov_b32_e32 v253, 0
	v_mov_b32_e32 v252, 0
	v_mov_b32_e32 v251, 0
	v_mov_b32_e32 v250, 0
	v_accvgpr_write_b32 a99, 0
	v_accvgpr_write_b32 a98, 0
	v_accvgpr_write_b32 a97, 0
	v_accvgpr_write_b32 a96, 0
	v_accvgpr_write_b32 a103, 0
	v_accvgpr_write_b32 a102, 0
	v_accvgpr_write_b32 a101, 0
	v_accvgpr_write_b32 a100, 0
	v_accvgpr_write_b32 a109, 0
	v_accvgpr_write_b32 a108, 0
	v_accvgpr_write_b32 a107, 0
	v_accvgpr_write_b32 a106, 0
	v_mov_b32_e32 v21, 0
	v_mov_b32_e32 v20, 0
	v_mov_b32_e32 v19, 0
	v_mov_b32_e32 v18, 0
	v_mov_b32_e32 v33, 0
	v_mov_b32_e32 v32, 0
	v_mov_b32_e32 v31, 0
	v_mov_b32_e32 v30, 0
	v_mov_b32_e32 v29, 0
	v_mov_b32_e32 v28, 0
	v_mov_b32_e32 v27, 0
	v_mov_b32_e32 v26, 0
	v_mov_b32_e32 v9, 0
	v_mov_b32_e32 v8, 0
	v_mov_b32_e32 v7, 0
	v_mov_b32_e32 v6, 0
	v_mov_b32_e32 v25, 0
	v_mov_b32_e32 v24, 0
	v_mov_b32_e32 v23, 0
	v_mov_b32_e32 v22, 0
	v_mov_b32_e32 v157, 0
	v_mov_b32_e32 v156, 0
	v_mov_b32_e32 v155, 0
	v_mov_b32_e32 v154, 0
	v_mov_b32_e32 v5, 0
	v_mov_b32_e32 v4, 0
	v_mov_b32_e32 v3, 0
	v_mov_b32_e32 v2, 0
.LBB0_5:                                ; %._crit_edge
	v_cvt_pk_f16_f32 v150, v2, v3
	v_cvt_pk_f16_f32 v151, v4, v5
	v_accvgpr_read_b32 v2, a106
	v_accvgpr_read_b32 v3, a107
	v_accvgpr_read_b32 v4, a108
	v_accvgpr_read_b32 v5, a109
	v_cvt_pk_f16_f32 v254, v2, v3
	v_cvt_pk_f16_f32 v255, v4, v5
	v_accvgpr_read_b32 v2, a100
	v_accvgpr_write_b32 a92, v250
	v_accvgpr_read_b32 v3, a101
	v_accvgpr_read_b32 v4, a102
	v_accvgpr_read_b32 v5, a103
	v_accvgpr_write_b32 a93, v251
	v_accvgpr_write_b32 a94, v252
	v_accvgpr_write_b32 a95, v253
	v_cvt_pk_f16_f32 v250, v2, v3
	v_cvt_pk_f16_f32 v251, v4, v5
	v_accvgpr_read_b32 v2, a96
	v_accvgpr_write_b32 a88, v246
	v_accvgpr_read_b32 v3, a97
	v_accvgpr_read_b32 v4, a98
	v_accvgpr_read_b32 v5, a99
	v_accvgpr_write_b32 a89, v247
	v_accvgpr_write_b32 a90, v248
	v_accvgpr_write_b32 a91, v249
	v_cvt_pk_f16_f32 v246, v2, v3
	v_cvt_pk_f16_f32 v247, v4, v5
	v_accvgpr_read_b32 v2, a92
	v_accvgpr_write_b32 a84, v242
	v_accvgpr_read_b32 v3, a93
	v_accvgpr_read_b32 v4, a94
	v_accvgpr_read_b32 v5, a95
	v_accvgpr_write_b32 a85, v243
	v_accvgpr_write_b32 a86, v244
	v_accvgpr_write_b32 a87, v245
	v_cvt_pk_f16_f32 v242, v2, v3
	v_cvt_pk_f16_f32 v243, v4, v5
	v_accvgpr_read_b32 v2, a88
	v_accvgpr_write_b32 a80, v238
	v_accvgpr_read_b32 v3, a89
	v_accvgpr_read_b32 v4, a90
	v_accvgpr_read_b32 v5, a91
	v_accvgpr_write_b32 a81, v239
	v_accvgpr_write_b32 a82, v240
	v_accvgpr_write_b32 a83, v241
	v_cvt_pk_f16_f32 v238, v2, v3
	v_cvt_pk_f16_f32 v239, v4, v5
	v_accvgpr_read_b32 v2, a84
	v_accvgpr_write_b32 a76, v234
	v_accvgpr_read_b32 v3, a85
	v_accvgpr_read_b32 v4, a86
	v_accvgpr_read_b32 v5, a87
	v_accvgpr_write_b32 a77, v235
	v_accvgpr_write_b32 a78, v236
	v_accvgpr_write_b32 a79, v237
	v_cvt_pk_f16_f32 v234, v2, v3
	v_cvt_pk_f16_f32 v235, v4, v5
	v_accvgpr_read_b32 v2, a80
	v_accvgpr_write_b32 a72, v230
	v_accvgpr_read_b32 v3, a81
	v_accvgpr_read_b32 v4, a82
	v_accvgpr_read_b32 v5, a83
	v_accvgpr_write_b32 a73, v231
	v_accvgpr_write_b32 a74, v232
	v_accvgpr_write_b32 a75, v233
	v_cvt_pk_f16_f32 v230, v2, v3
	v_cvt_pk_f16_f32 v231, v4, v5
	v_accvgpr_read_b32 v2, a76
	v_accvgpr_write_b32 a68, v226
	v_accvgpr_read_b32 v3, a77
	v_accvgpr_read_b32 v4, a78
	v_accvgpr_read_b32 v5, a79
	v_accvgpr_write_b32 a69, v227
	v_accvgpr_write_b32 a70, v228
	v_accvgpr_write_b32 a71, v229
	v_cvt_pk_f16_f32 v226, v2, v3
	v_cvt_pk_f16_f32 v227, v4, v5
	v_accvgpr_read_b32 v2, a72
	v_accvgpr_write_b32 a64, v222
	v_accvgpr_read_b32 v3, a73
	v_accvgpr_read_b32 v4, a74
	v_accvgpr_read_b32 v5, a75
	v_accvgpr_write_b32 a65, v223
	v_accvgpr_write_b32 a66, v224
	v_accvgpr_write_b32 a67, v225
	v_cvt_pk_f16_f32 v222, v2, v3
	v_cvt_pk_f16_f32 v223, v4, v5
	v_accvgpr_read_b32 v2, a68
	v_accvgpr_write_b32 a60, v218
	v_accvgpr_read_b32 v3, a69
	v_accvgpr_read_b32 v4, a70
	v_accvgpr_read_b32 v5, a71
	v_accvgpr_write_b32 a61, v219
	v_accvgpr_write_b32 a62, v220
	v_accvgpr_write_b32 a63, v221
	v_cvt_pk_f16_f32 v218, v2, v3
	v_cvt_pk_f16_f32 v219, v4, v5
	v_accvgpr_read_b32 v2, a64
	v_accvgpr_write_b32 a56, v214
	v_accvgpr_read_b32 v3, a65
	v_accvgpr_read_b32 v4, a66
	v_accvgpr_read_b32 v5, a67
	v_accvgpr_write_b32 a57, v215
	v_accvgpr_write_b32 a58, v216
	v_accvgpr_write_b32 a59, v217
	v_cvt_pk_f16_f32 v214, v2, v3
	v_cvt_pk_f16_f32 v215, v4, v5
	v_accvgpr_read_b32 v2, a60
	v_accvgpr_write_b32 a52, v210
	v_accvgpr_read_b32 v3, a61
	v_accvgpr_read_b32 v4, a62
	v_accvgpr_read_b32 v5, a63
	v_accvgpr_write_b32 a53, v211
	v_accvgpr_write_b32 a54, v212
	v_accvgpr_write_b32 a55, v213
	v_cvt_pk_f16_f32 v210, v2, v3
	v_cvt_pk_f16_f32 v211, v4, v5
	v_accvgpr_read_b32 v2, a56
	v_accvgpr_write_b32 a48, v206
	v_accvgpr_read_b32 v3, a57
	v_accvgpr_read_b32 v4, a58
	v_accvgpr_read_b32 v5, a59
	v_accvgpr_write_b32 a49, v207
	v_accvgpr_write_b32 a50, v208
	v_accvgpr_write_b32 a51, v209
	v_cvt_pk_f16_f32 v206, v2, v3
	v_cvt_pk_f16_f32 v207, v4, v5
	v_accvgpr_read_b32 v2, a52
	v_accvgpr_write_b32 a44, v202
	v_accvgpr_read_b32 v3, a53
	v_accvgpr_read_b32 v4, a54
	v_accvgpr_read_b32 v5, a55
	v_accvgpr_write_b32 a45, v203
	v_accvgpr_write_b32 a46, v204
	v_accvgpr_write_b32 a47, v205
	v_cvt_pk_f16_f32 v202, v2, v3
	v_cvt_pk_f16_f32 v203, v4, v5
	v_accvgpr_read_b32 v2, a48
	v_accvgpr_read_b32 v3, a49
	v_accvgpr_read_b32 v4, a50
	v_accvgpr_read_b32 v5, a51
	v_cvt_pk_f16_f32 v198, v2, v3
	v_cvt_pk_f16_f32 v199, v4, v5
	v_accvgpr_read_b32 v2, a44
	v_accvgpr_read_b32 v3, a45
	v_accvgpr_read_b32 v4, a46
	v_accvgpr_read_b32 v5, a47
	v_cvt_pk_f16_f32 v194, v2, v3
	v_cvt_pk_f16_f32 v195, v4, v5
	v_accvgpr_read_b32 v2, a252
	v_accvgpr_read_b32 v3, a253
	v_accvgpr_read_b32 v4, a254
	v_accvgpr_read_b32 v5, a255
	v_cvt_pk_f16_f32 v190, v2, v3
	v_cvt_pk_f16_f32 v191, v4, v5
	v_accvgpr_read_b32 v2, a248
	v_accvgpr_read_b32 v3, a249
	v_accvgpr_read_b32 v4, a250
	v_accvgpr_read_b32 v5, a251
	v_cvt_pk_f16_f32 v186, v2, v3
	v_cvt_pk_f16_f32 v187, v4, v5
	v_accvgpr_read_b32 v2, a244
	v_accvgpr_read_b32 v3, a245
	v_accvgpr_read_b32 v4, a246
	v_accvgpr_read_b32 v5, a247
	v_cvt_pk_f16_f32 v182, v2, v3
	v_cvt_pk_f16_f32 v183, v4, v5
	v_accvgpr_read_b32 v2, a240
	v_accvgpr_read_b32 v3, a241
	v_accvgpr_read_b32 v4, a242
	v_accvgpr_read_b32 v5, a243
	v_cvt_pk_f16_f32 v178, v2, v3
	v_cvt_pk_f16_f32 v179, v4, v5
	v_accvgpr_read_b32 v2, a236
	v_accvgpr_read_b32 v3, a237
	v_accvgpr_read_b32 v4, a238
	v_accvgpr_read_b32 v5, a239
	v_cvt_pk_f16_f32 v128, v2, v3
	v_cvt_pk_f16_f32 v129, v4, v5
	v_accvgpr_read_b32 v2, a220
	v_accvgpr_read_b32 v3, a221
	v_accvgpr_read_b32 v4, a222
	v_accvgpr_read_b32 v5, a223
	v_cvt_pk_f16_f32 v170, v2, v3
	v_cvt_pk_f16_f32 v171, v4, v5
	v_accvgpr_read_b32 v2, a12
	v_accvgpr_read_b32 v3, a13
	v_accvgpr_read_b32 v4, a14
	v_accvgpr_read_b32 v5, a15
	v_cvt_pk_f16_f32 v140, v2, v3
	v_cvt_pk_f16_f32 v141, v4, v5
	v_accvgpr_read_b32 v2, a114
	v_accvgpr_read_b32 v3, a115
	v_accvgpr_read_b32 v4, a116
	v_accvgpr_read_b32 v5, a117
	v_cvt_pk_f16_f32 v14, v22, v23
	v_cvt_pk_f16_f32 v22, v2, v3
	v_cvt_pk_f16_f32 v23, v4, v5
	v_accvgpr_read_b32 v2, a16
	v_accvgpr_read_b32 v3, a17
	v_accvgpr_read_b32 v4, a18
	v_accvgpr_read_b32 v5, a19
	v_cvt_pk_f16_f32 v12, v6, v7
	v_cvt_pk_f16_f32 v7, v20, v21
	v_cvt_pk_f16_f32 v21, v40, v41
	v_cvt_pk_f16_f32 v40, v2, v3
	v_cvt_pk_f16_f32 v41, v4, v5
	v_accvgpr_read_b32 v2, a4
	v_accvgpr_read_b32 v3, a5
	v_accvgpr_read_b32 v4, a6
	v_accvgpr_read_b32 v5, a7
	v_cvt_pk_f16_f32 v13, v8, v9
	v_cvt_pk_f16_f32 v9, v32, v33
	v_cvt_pk_f16_f32 v6, v18, v19
	v_cvt_pk_f16_f32 v19, v36, v37
	v_cvt_pk_f16_f32 v36, v2, v3
	v_cvt_pk_f16_f32 v37, v4, v5
	v_accvgpr_read_b32 v5, a3
	v_cvt_pk_f16_f32 v32, v42, v43
	v_cvt_pk_f16_f32 v33, v44, v45
	v_accvgpr_read_b32 v45, a11
	v_accvgpr_read_b32 v3, a1
	v_accvgpr_read_b32 v2, a0
	v_accvgpr_read_b32 v43, a9
	v_accvgpr_read_b32 v42, a8
	v_cvt_pk_f16_f32 v18, v34, v35
	v_cvt_pk_f16_f32 v34, v2, v3
	v_cvt_pk_f16_f32 v2, v42, v43
	v_and_b32_e32 v42, 15, v0
	s_lshr_b32 s0, s15, 3
	v_and_or_b32 v42, s0, 16, v42
	s_mul_i32 s0, s16, s13
	s_ashr_i32 s1, s0, 31
	s_lshl_b64 s[0:1], s[0:1], 1
	s_add_u32 s2, s6, s0
	v_accvgpr_read_b32 v62, a204
	v_accvgpr_read_b32 v58, a200
	v_accvgpr_read_b32 v54, a172
	v_lshrrev_b32_e32 v50, 2, v50
	s_addc_u32 s3, s7, s1
	s_ashr_i32 s15, s14, 31
	v_accvgpr_read_b32 v74, a164
	v_accvgpr_read_b32 v63, a205
	v_accvgpr_read_b32 v64, a206
	v_accvgpr_read_b32 v65, a207
	v_accvgpr_read_b32 v59, a201
	v_accvgpr_read_b32 v60, a202
	v_accvgpr_read_b32 v61, a203
	v_accvgpr_read_b32 v55, a173
	v_accvgpr_read_b32 v56, a174
	v_accvgpr_read_b32 v57, a175
	v_accvgpr_read_b32 v44, a10
	v_and_b32_e32 v50, 28, v50
	s_lshl_b64 s[0:1], s[14:15], 1
	v_cvt_pk_f16_f32 v15, v24, v25
	v_accvgpr_read_b32 v75, a165
	v_accvgpr_read_b32 v76, a166
	v_accvgpr_read_b32 v77, a167
	v_cvt_pk_f16_f32 v62, v62, v63
	v_cvt_pk_f16_f32 v63, v64, v65
	v_cvt_pk_f16_f32 v58, v58, v59
	v_cvt_pk_f16_f32 v59, v60, v61
	v_cvt_pk_f16_f32 v54, v54, v55
	v_cvt_pk_f16_f32 v55, v56, v57
	v_cvt_pk_f16_f32 v24, v46, v47
	v_cvt_pk_f16_f32 v25, v48, v49
	v_cvt_pk_f16_f32 v3, v44, v45
	v_or_b32_e32 v43, 32, v42
	v_or_b32_e32 v44, 64, v42
	v_or_b32_e32 v45, 0x60, v42
	v_or_b32_e32 v46, 0x80, v42
	v_or_b32_e32 v47, 0xa0, v42
	v_or_b32_e32 v48, 0xc0, v42
	v_or_b32_e32 v49, 0xe0, v42
	v_or_b32_e32 v51, 32, v50
	v_or_b32_e32 v52, 64, v50
	v_or_b32_e32 v53, 0x60, v50
	v_or_b32_e32 v56, 0x80, v50
	v_or_b32_e32 v57, 0xa0, v50
	v_or_b32_e32 v60, 0xc0, v50
	v_or_b32_e32 v61, 0xe0, v50
	s_add_u32 s36, s2, s0
	v_mul_lo_u32 v64, v42, s13
	v_cmp_gt_i32_e64 s[28:29], s8, v42
	v_cmp_gt_i32_e64 s[14:15], s9, v50
	v_cvt_pk_f16_f32 v74, v74, v75
	v_cvt_pk_f16_f32 v75, v76, v77
	v_cvt_pk_f16_f32 v70, v70, v71
	v_cvt_pk_f16_f32 v71, v72, v73
	v_cvt_pk_f16_f32 v66, v66, v67
	v_cvt_pk_f16_f32 v67, v68, v69
	s_addc_u32 s33, s3, s1
	v_mul_lo_u32 v65, v43, s13
	v_mul_lo_u32 v68, v44, s13
	v_mul_lo_u32 v69, v45, s13
	v_mul_lo_u32 v72, v46, s13
	v_mul_lo_u32 v73, v47, s13
	v_mul_lo_u32 v76, v48, s13
	v_mul_lo_u32 v77, v49, s13
	v_cmp_gt_i32_e64 s[26:27], s8, v43
	v_cmp_gt_i32_e64 s[24:25], s8, v44
	v_cmp_gt_i32_e64 s[22:23], s8, v45
	v_cmp_gt_i32_e64 s[20:21], s8, v46
	v_cmp_gt_i32_e64 s[18:19], s8, v47
	v_cmp_gt_i32_e64 s[16:17], s8, v48
	v_cmp_gt_i32_e32 vcc, s8, v49
	v_cmp_gt_i32_e64 s[12:13], s9, v51
	v_cmp_gt_i32_e64 s[10:11], s9, v52
	v_cmp_gt_i32_e64 s[30:31], s9, v53
	v_cmp_gt_i32_e64 s[6:7], s9, v56
	v_cmp_gt_i32_e64 s[4:5], s9, v57
	v_cmp_gt_i32_e64 s[2:3], s9, v60
	v_cmp_gt_i32_e64 s[0:1], s9, v61
	v_add_lshl_u32 v42, v64, v50, 1
	v_bfrev_b32_e32 v43, 1
	s_and_b64 s[8:9], s[28:29], s[14:15]
	v_cndmask_b32_e64 v42, v43, v42, s[8:9]
	v_add_lshl_u32 v0, v51, v64, 1
	s_and_b64 s[8:9], s[28:29], s[12:13]
	v_cvt_pk_f16_f32 v16, v154, v155
	v_cvt_pk_f16_f32 v17, v156, v157
	s_and_b32 s37, s33, 0xffff
	s_mov_b32 s39, 0x27000
	s_mov_b32 s38, 0x7ffffffe
	v_cndmask_b32_e64 v0, v43, v0, s[8:9]
	buffer_store_dwordx2 v[150:151], v42, s[36:39], 0 offen
	buffer_store_dwordx2 v[16:17], v0, s[36:39], 0 offen
	v_add_lshl_u32 v0, v52, v64, 1
	s_and_b64 s[8:9], s[28:29], s[10:11]
	v_cndmask_b32_e64 v0, v43, v0, s[8:9]
	buffer_store_dwordx2 v[14:15], v0, s[36:39], 0 offen
	v_add_lshl_u32 v0, v53, v64, 1
	s_and_b64 s[8:9], s[28:29], s[30:31]
	v_cndmask_b32_e64 v0, v43, v0, s[8:9]
	buffer_store_dwordx2 v[12:13], v0, s[36:39], 0 offen
	v_add_lshl_u32 v0, v56, v64, 1
	s_and_b64 s[8:9], s[28:29], s[6:7]
	v_cvt_pk_f16_f32 v10, v26, v27
	v_cvt_pk_f16_f32 v11, v28, v29
	v_cndmask_b32_e64 v0, v43, v0, s[8:9]
	buffer_store_dwordx2 v[10:11], v0, s[36:39], 0 offen
	v_add_lshl_u32 v0, v57, v64, 1
	s_and_b64 s[8:9], s[28:29], s[4:5]
	v_cvt_pk_f16_f32 v8, v30, v31
	v_cndmask_b32_e64 v0, v43, v0, s[8:9]
	buffer_store_dwordx2 v[8:9], v0, s[36:39], 0 offen
	v_add_lshl_u32 v0, v60, v64, 1
	s_and_b64 s[8:9], s[28:29], s[2:3]
	v_cndmask_b32_e64 v0, v43, v0, s[8:9]
	buffer_store_dwordx2 v[6:7], v0, s[36:39], 0 offen
	v_add_lshl_u32 v0, v61, v64, 1
	s_and_b64 s[8:9], s[28:29], s[0:1]
	v_cndmask_b32_e64 v0, v43, v0, s[8:9]
	buffer_store_dwordx2 v[254:255], v0, s[36:39], 0 offen
	v_add_lshl_u32 v0, v65, v50, 1
	s_and_b64 s[8:9], s[26:27], s[14:15]
	v_cndmask_b32_e64 v0, v43, v0, s[8:9]
	buffer_store_dwordx2 v[250:251], v0, s[36:39], 0 offen
	v_add_lshl_u32 v0, v65, v51, 1
	s_and_b64 s[8:9], s[26:27], s[12:13]
	v_cndmask_b32_e64 v0, v43, v0, s[8:9]
	buffer_store_dwordx2 v[246:247], v0, s[36:39], 0 offen
	v_add_lshl_u32 v0, v65, v52, 1
	s_and_b64 s[8:9], s[26:27], s[10:11]
	v_cndmask_b32_e64 v0, v43, v0, s[8:9]
	buffer_store_dwordx2 v[242:243], v0, s[36:39], 0 offen
	v_add_lshl_u32 v0, v65, v53, 1
	s_and_b64 s[8:9], s[26:27], s[30:31]
	v_cndmask_b32_e64 v0, v43, v0, s[8:9]
	buffer_store_dwordx2 v[238:239], v0, s[36:39], 0 offen
	v_add_lshl_u32 v0, v65, v56, 1
	s_and_b64 s[8:9], s[26:27], s[6:7]
	v_cndmask_b32_e64 v0, v43, v0, s[8:9]
	buffer_store_dwordx2 v[234:235], v0, s[36:39], 0 offen
	v_add_lshl_u32 v0, v65, v57, 1
	s_and_b64 s[8:9], s[26:27], s[4:5]
	v_cndmask_b32_e64 v0, v43, v0, s[8:9]
	buffer_store_dwordx2 v[230:231], v0, s[36:39], 0 offen
	v_add_lshl_u32 v0, v65, v60, 1
	s_and_b64 s[8:9], s[26:27], s[2:3]
	v_cndmask_b32_e64 v0, v43, v0, s[8:9]
	buffer_store_dwordx2 v[226:227], v0, s[36:39], 0 offen
	v_add_lshl_u32 v0, v65, v61, 1
	s_and_b64 s[8:9], s[26:27], s[0:1]
	v_cndmask_b32_e64 v0, v43, v0, s[8:9]
	buffer_store_dwordx2 v[222:223], v0, s[36:39], 0 offen
	v_add_lshl_u32 v0, v68, v50, 1
	s_and_b64 s[8:9], s[24:25], s[14:15]
	v_cndmask_b32_e64 v0, v43, v0, s[8:9]
	buffer_store_dwordx2 v[218:219], v0, s[36:39], 0 offen
	v_add_lshl_u32 v0, v68, v51, 1
	s_and_b64 s[8:9], s[24:25], s[12:13]
	v_cndmask_b32_e64 v0, v43, v0, s[8:9]
	buffer_store_dwordx2 v[214:215], v0, s[36:39], 0 offen
	v_add_lshl_u32 v0, v68, v52, 1
	s_and_b64 s[8:9], s[24:25], s[10:11]
	v_cndmask_b32_e64 v0, v43, v0, s[8:9]
	buffer_store_dwordx2 v[210:211], v0, s[36:39], 0 offen
	v_add_lshl_u32 v0, v68, v53, 1
	s_and_b64 s[8:9], s[24:25], s[30:31]
	v_cndmask_b32_e64 v0, v43, v0, s[8:9]
	buffer_store_dwordx2 v[206:207], v0, s[36:39], 0 offen
	v_add_lshl_u32 v0, v68, v56, 1
	s_and_b64 s[8:9], s[24:25], s[6:7]
	v_cndmask_b32_e64 v0, v43, v0, s[8:9]
	buffer_store_dwordx2 v[202:203], v0, s[36:39], 0 offen
	v_add_lshl_u32 v0, v68, v57, 1
	s_and_b64 s[8:9], s[24:25], s[4:5]
	v_cndmask_b32_e64 v0, v43, v0, s[8:9]
	buffer_store_dwordx2 v[198:199], v0, s[36:39], 0 offen
	v_add_lshl_u32 v0, v68, v60, 1
	s_and_b64 s[8:9], s[24:25], s[2:3]
	v_cndmask_b32_e64 v0, v43, v0, s[8:9]
	buffer_store_dwordx2 v[194:195], v0, s[36:39], 0 offen
	v_add_lshl_u32 v0, v68, v61, 1
	s_and_b64 s[8:9], s[24:25], s[0:1]
	v_cndmask_b32_e64 v0, v43, v0, s[8:9]
	buffer_store_dwordx2 v[190:191], v0, s[36:39], 0 offen
	v_add_lshl_u32 v0, v69, v50, 1
	s_and_b64 s[8:9], s[22:23], s[14:15]
	v_cndmask_b32_e64 v0, v43, v0, s[8:9]
	buffer_store_dwordx2 v[186:187], v0, s[36:39], 0 offen
	v_add_lshl_u32 v0, v69, v51, 1
	s_and_b64 s[8:9], s[22:23], s[12:13]
	v_cndmask_b32_e64 v0, v43, v0, s[8:9]
	buffer_store_dwordx2 v[182:183], v0, s[36:39], 0 offen
	v_add_lshl_u32 v0, v69, v52, 1
	s_and_b64 s[8:9], s[22:23], s[10:11]
	v_cndmask_b32_e64 v0, v43, v0, s[8:9]
	buffer_store_dwordx2 v[178:179], v0, s[36:39], 0 offen
	v_add_lshl_u32 v0, v69, v53, 1
	s_and_b64 s[8:9], s[22:23], s[30:31]
	v_cndmask_b32_e64 v0, v43, v0, s[8:9]
	buffer_store_dwordx2 v[128:129], v0, s[36:39], 0 offen
	v_add_lshl_u32 v0, v69, v56, 1
	s_and_b64 s[8:9], s[22:23], s[6:7]
	v_cndmask_b32_e64 v0, v43, v0, s[8:9]
	buffer_store_dwordx2 v[170:171], v0, s[36:39], 0 offen
	v_add_lshl_u32 v0, v69, v57, 1
	s_and_b64 s[8:9], s[22:23], s[4:5]
	v_accvgpr_read_b32 v174, a216
	v_cndmask_b32_e64 v0, v43, v0, s[8:9]
	v_accvgpr_read_b32 v175, a217
	v_accvgpr_read_b32 v176, a218
	v_accvgpr_read_b32 v177, a219
	buffer_store_dwordx2 v[140:141], v0, s[36:39], 0 offen
	v_add_lshl_u32 v0, v69, v60, 1
	s_and_b64 s[8:9], s[22:23], s[2:3]
	v_cvt_pk_f16_f32 v154, v174, v175
	v_cvt_pk_f16_f32 v155, v176, v177
	v_cndmask_b32_e64 v0, v43, v0, s[8:9]
	buffer_store_dwordx2 v[154:155], v0, s[36:39], 0 offen
	v_add_lshl_u32 v0, v69, v61, 1
	s_and_b64 s[8:9], s[22:23], s[0:1]
	v_cvt_pk_f16_f32 v142, v132, v133
	v_cvt_pk_f16_f32 v143, v134, v135
	v_cndmask_b32_e64 v0, v43, v0, s[8:9]
	buffer_store_dwordx2 v[142:143], v0, s[36:39], 0 offen
	v_add_lshl_u32 v0, v72, v50, 1
	s_and_b64 s[8:9], s[20:21], s[14:15]
	v_cvt_pk_f16_f32 v134, v136, v137
	v_cvt_pk_f16_f32 v135, v138, v139
	v_cndmask_b32_e64 v0, v43, v0, s[8:9]
	buffer_store_dwordx2 v[134:135], v0, s[36:39], 0 offen
	v_add_lshl_u32 v0, v72, v51, 1
	s_and_b64 s[8:9], s[20:21], s[12:13]
	v_cvt_pk_f16_f32 v126, v144, v145
	v_cvt_pk_f16_f32 v127, v146, v147
	v_cndmask_b32_e64 v0, v43, v0, s[8:9]
	buffer_store_dwordx2 v[126:127], v0, s[36:39], 0 offen
	v_add_lshl_u32 v0, v72, v52, 1
	s_and_b64 s[8:9], s[20:21], s[10:11]
	v_cvt_pk_f16_f32 v118, v118, v119
	v_cvt_pk_f16_f32 v119, v120, v121
	v_cndmask_b32_e64 v0, v43, v0, s[8:9]
	buffer_store_dwordx2 v[118:119], v0, s[36:39], 0 offen
	v_add_lshl_u32 v0, v72, v53, 1
	s_and_b64 s[8:9], s[20:21], s[30:31]
	v_cvt_pk_f16_f32 v110, v110, v111
	v_cvt_pk_f16_f32 v111, v112, v113
	v_cndmask_b32_e64 v0, v43, v0, s[8:9]
	buffer_store_dwordx2 v[110:111], v0, s[36:39], 0 offen
	v_add_lshl_u32 v0, v72, v56, 1
	s_and_b64 s[8:9], s[20:21], s[6:7]
	v_cvt_pk_f16_f32 v106, v106, v107
	v_cvt_pk_f16_f32 v107, v108, v109
	v_cndmask_b32_e64 v0, v43, v0, s[8:9]
	buffer_store_dwordx2 v[106:107], v0, s[36:39], 0 offen
	v_add_lshl_u32 v0, v72, v57, 1
	s_and_b64 s[8:9], s[20:21], s[4:5]
	v_cvt_pk_f16_f32 v102, v102, v103
	v_cvt_pk_f16_f32 v103, v104, v105
	v_cndmask_b32_e64 v0, v43, v0, s[8:9]
	buffer_store_dwordx2 v[102:103], v0, s[36:39], 0 offen
	v_add_lshl_u32 v0, v72, v60, 1
	s_and_b64 s[8:9], s[20:21], s[2:3]
	v_cvt_pk_f16_f32 v98, v98, v99
	v_cvt_pk_f16_f32 v99, v100, v101
	v_cndmask_b32_e64 v0, v43, v0, s[8:9]
	buffer_store_dwordx2 v[98:99], v0, s[36:39], 0 offen
	v_add_lshl_u32 v0, v72, v61, 1
	s_and_b64 s[8:9], s[20:21], s[0:1]
	v_cvt_pk_f16_f32 v94, v94, v95
	v_cvt_pk_f16_f32 v95, v96, v97
	v_accvgpr_read_b32 v90, a168
	v_cndmask_b32_e64 v0, v43, v0, s[8:9]
	v_accvgpr_read_b32 v91, a169
	v_accvgpr_read_b32 v92, a170
	v_accvgpr_read_b32 v93, a171
	buffer_store_dwordx2 v[94:95], v0, s[36:39], 0 offen
	v_add_lshl_u32 v0, v73, v50, 1
	s_and_b64 s[8:9], s[18:19], s[14:15]
	v_cvt_pk_f16_f32 v90, v90, v91
	v_cvt_pk_f16_f32 v91, v92, v93
	v_accvgpr_read_b32 v86, a110
	v_cndmask_b32_e64 v0, v43, v0, s[8:9]
	v_accvgpr_read_b32 v87, a111
	v_accvgpr_read_b32 v88, a112
	v_accvgpr_read_b32 v89, a113
	buffer_store_dwordx2 v[90:91], v0, s[36:39], 0 offen
	v_add_lshl_u32 v0, v73, v51, 1
	s_and_b64 s[8:9], s[18:19], s[12:13]
	v_cvt_pk_f16_f32 v86, v86, v87
	v_cvt_pk_f16_f32 v87, v88, v89
	v_cndmask_b32_e64 v0, v43, v0, s[8:9]
	buffer_store_dwordx2 v[86:87], v0, s[36:39], 0 offen
	v_add_lshl_u32 v0, v73, v52, 1
	s_and_b64 s[8:9], s[18:19], s[10:11]
	v_cvt_pk_f16_f32 v82, v82, v83
	v_cvt_pk_f16_f32 v83, v84, v85
	v_cndmask_b32_e64 v0, v43, v0, s[8:9]
	buffer_store_dwordx2 v[82:83], v0, s[36:39], 0 offen
	v_add_lshl_u32 v0, v73, v53, 1
	s_and_b64 s[8:9], s[18:19], s[30:31]
	v_cvt_pk_f16_f32 v78, v78, v79
	v_cvt_pk_f16_f32 v79, v80, v81
	v_cndmask_b32_e64 v0, v43, v0, s[8:9]
	buffer_store_dwordx2 v[78:79], v0, s[36:39], 0 offen
	v_add_lshl_u32 v0, v73, v56, 1
	s_and_b64 s[8:9], s[18:19], s[6:7]
	v_cndmask_b32_e64 v0, v43, v0, s[8:9]
	buffer_store_dwordx2 v[74:75], v0, s[36:39], 0 offen
	v_add_lshl_u32 v0, v73, v57, 1
	s_and_b64 s[8:9], s[18:19], s[4:5]
	v_cndmask_b32_e64 v0, v43, v0, s[8:9]
	buffer_store_dwordx2 v[70:71], v0, s[36:39], 0 offen
	v_add_lshl_u32 v0, v73, v60, 1
	s_and_b64 s[8:9], s[18:19], s[2:3]
	v_cndmask_b32_e64 v0, v43, v0, s[8:9]
	buffer_store_dwordx2 v[66:67], v0, s[36:39], 0 offen
	v_add_lshl_u32 v0, v73, v61, 1
	s_and_b64 s[8:9], s[18:19], s[0:1]
	v_cndmask_b32_e64 v0, v43, v0, s[8:9]
	buffer_store_dwordx2 v[62:63], v0, s[36:39], 0 offen
	v_add_lshl_u32 v0, v76, v50, 1
	s_and_b64 s[8:9], s[16:17], s[14:15]
	v_cndmask_b32_e64 v0, v43, v0, s[8:9]
	buffer_store_dwordx2 v[58:59], v0, s[36:39], 0 offen
	v_add_lshl_u32 v0, v76, v51, 1
	s_and_b64 s[8:9], s[16:17], s[12:13]
	v_cndmask_b32_e64 v0, v43, v0, s[8:9]
	buffer_store_dwordx2 v[54:55], v0, s[36:39], 0 offen
	v_add_lshl_u32 v0, v76, v52, 1
	s_and_b64 s[8:9], s[16:17], s[10:11]
	v_cvt_pk_f16_f32 v28, v122, v123
	v_cvt_pk_f16_f32 v29, v124, v125
	v_cndmask_b32_e64 v0, v43, v0, s[8:9]
	buffer_store_dwordx2 v[28:29], v0, s[36:39], 0 offen
	v_add_lshl_u32 v0, v76, v53, 1
	s_and_b64 s[8:9], s[16:17], s[30:31]
	v_cvt_pk_f16_f32 v26, v114, v115
	v_cvt_pk_f16_f32 v27, v116, v117
	v_cndmask_b32_e64 v0, v43, v0, s[8:9]
	buffer_store_dwordx2 v[26:27], v0, s[36:39], 0 offen
	v_add_lshl_u32 v0, v76, v56, 1
	s_and_b64 s[8:9], s[16:17], s[6:7]
	v_cndmask_b32_e64 v0, v43, v0, s[8:9]
	buffer_store_dwordx2 v[24:25], v0, s[36:39], 0 offen
	v_add_lshl_u32 v0, v76, v57, 1
	s_and_b64 s[8:9], s[16:17], s[4:5]
	v_cndmask_b32_e64 v0, v43, v0, s[8:9]
	buffer_store_dwordx2 v[22:23], v0, s[36:39], 0 offen
	v_add_lshl_u32 v0, v76, v60, 1
	s_and_b64 s[8:9], s[16:17], s[2:3]
	v_cvt_pk_f16_f32 v20, v38, v39
	v_cndmask_b32_e64 v0, v43, v0, s[8:9]
	buffer_store_dwordx2 v[20:21], v0, s[36:39], 0 offen
	v_add_lshl_u32 v0, v76, v61, 1
	s_and_b64 s[8:9], s[16:17], s[0:1]
	v_cndmask_b32_e64 v0, v43, v0, s[8:9]
	buffer_store_dwordx2 v[18:19], v0, s[36:39], 0 offen
	v_add_lshl_u32 v0, v77, v50, 1
	s_and_b64 s[8:9], vcc, s[14:15]
	v_cndmask_b32_e64 v0, v43, v0, s[8:9]
	buffer_store_dwordx2 v[40:41], v0, s[36:39], 0 offen
	v_add_lshl_u32 v0, v77, v51, 1
	s_and_b64 s[8:9], vcc, s[12:13]
	v_cvt_pk_f16_f32 v38, v162, v163
	v_cvt_pk_f16_f32 v39, v164, v165
	v_cndmask_b32_e64 v0, v43, v0, s[8:9]
	buffer_store_dwordx2 v[38:39], v0, s[36:39], 0 offen
	v_add_lshl_u32 v0, v77, v52, 1
	s_and_b64 s[8:9], vcc, s[10:11]
	v_cndmask_b32_e64 v0, v43, v0, s[8:9]
	v_accvgpr_read_b32 v4, a2
	buffer_store_dwordx2 v[36:37], v0, s[36:39], 0 offen
	v_add_lshl_u32 v0, v77, v53, 1
	s_and_b64 s[8:9], vcc, s[30:31]
	v_cvt_pk_f16_f32 v35, v4, v5
	v_cndmask_b32_e64 v0, v43, v0, s[8:9]
	buffer_store_dwordx2 v[34:35], v0, s[36:39], 0 offen
	v_add_lshl_u32 v0, v77, v56, 1
	s_and_b64 s[6:7], vcc, s[6:7]
	v_cndmask_b32_e64 v0, v43, v0, s[6:7]
	buffer_store_dwordx2 v[32:33], v0, s[36:39], 0 offen
	v_add_lshl_u32 v0, v77, v57, 1
	s_and_b64 s[4:5], vcc, s[4:5]
	v_cvt_pk_f16_f32 v30, v158, v159
	v_cvt_pk_f16_f32 v31, v160, v161
	v_cndmask_b32_e64 v0, v43, v0, s[4:5]
	buffer_store_dwordx2 v[30:31], v0, s[36:39], 0 offen
	v_add_lshl_u32 v0, v77, v60, 1
	s_and_b64 s[2:3], vcc, s[2:3]
	v_cvt_pk_f16_f32 v4, v166, v167
	v_cvt_pk_f16_f32 v5, v168, v169
	v_cndmask_b32_e64 v0, v43, v0, s[2:3]
	buffer_store_dwordx2 v[4:5], v0, s[36:39], 0 offen
	v_add_lshl_u32 v0, v77, v61, 1
	s_and_b64 vcc, vcc, s[0:1]
	v_cndmask_b32_e32 v0, v43, v0, vcc
	buffer_store_dwordx2 v[2:3], v0, s[36:39], 0 offen
	s_endpgm
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel v1_buffer_load
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
	.size	v1_buffer_load, .Lfunc_end0-v1_buffer_load
	.cfi_endproc
                                        ; -- End function
	.set v1_buffer_load.num_vgpr, 256
	.set v1_buffer_load.num_agpr, 256
	.set v1_buffer_load.numbered_sgpr, 40
	.set v1_buffer_load.num_named_barrier, 0
	.set v1_buffer_load.private_seg_size, 0
	.set v1_buffer_load.uses_vcc, 1
	.set v1_buffer_load.uses_flat_scratch, 0
	.set v1_buffer_load.has_dyn_sized_stack, 0
	.set v1_buffer_load.has_recursion, 0
	.set v1_buffer_load.has_indirect_call, 0
	.section	.AMDGPU.csdata,"",@progbits
; Kernel info:
; codeLenInByte = 12812
; TotalNumSgprs: 46
; NumVgprs: 256
; NumAgprs: 256
; TotalNumVgprs: 512
; ScratchSize: 0
; MemoryBound: 0
; FloatMode: 240
; IeeeMode: 1
; LDSByteSize: 0 bytes/workgroup (compile time only)
; SGPRBlocks: 5
; VGPRBlocks: 63
; NumSGPRsForWavesPerEU: 46
; NumVGPRsForWavesPerEU: 512
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
	.byte	16                              ; DW_AT_call_line
	.byte	27                              ; DW_AT_call_column
	.byte	5                               ; Abbrev [5] 0x55:0xc DW_TAG_inlined_subroutine
	.long	42                              ; DW_AT_abstract_origin
	.long	.Ldebug_ranges0                 ; DW_AT_ranges
	.byte	1                               ; DW_AT_call_file
	.byte	60                              ; DW_AT_call_line
	.byte	33                              ; DW_AT_call_column
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
	.asciz	"triton"                        ; string offset=0
.Linfo_string1:
	.asciz	"matmul_kernel.py"              ; string offset=7
.Linfo_string2:
	.asciz	"/root/gfx9-gluon-tutorials/kernels/gemm/a16w16/v1_buffer_load" ; string offset=24
.Linfo_string3:
	.asciz	"v1_buffer_load"                ; string offset=86
	.section	".note.GNU-stack","",@progbits
	.amdgpu_metadata
---
amdhsa.kernels:
  - .agpr_count:     256
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
    .name:           v1_buffer_load
    .private_segment_fixed_size: 0
    .sgpr_count:     46
    .sgpr_spill_count: 0
    .symbol:         v1_buffer_load.kd
    .uniform_work_group_size: 1
    .uses_dynamic_stack: false
    .vgpr_count:     512
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
