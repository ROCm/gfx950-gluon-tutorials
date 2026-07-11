	.amdgcn_target "amdgcn-amd-amdhsa--gfx950"
	.amdhsa_code_object_version 5
	.text
	.globl	v3_lds_swizzling                ; -- Begin function v3_lds_swizzling
	.p2align	8
	.type	v3_lds_swizzling,@function
v3_lds_swizzling:                       ; @v3_lds_swizzling
.Lfunc_begin0:
	.cfi_sections .debug_frame
	.cfi_startproc
; %bb.3:
	.file	1 "kernels/gemm/a16w16/v3_lds" "matmul_kernel.py"
	s_load_dwordx8 s[8:15], s[4:5], 0x0
	s_waitcnt lgkmcnt(0)
	s_branch .LBB0_0
	.p2align	8
; %bb.4:
.LBB0_0:                                ; %.lr.ph
	.file	2 "python/triton/language" "standard.py"
	s_add_i32 s0, s15, 0xff
	s_ashr_i32 s1, s0, 31
	s_lshr_b32 s1, s1, 24
	s_add_i32 s0, s0, s1
	s_ashr_i32 s6, s0, 8
	s_abs_i32 s7, s6
	v_cvt_f32_u32_e32 v1, s7
	s_sub_i32 s18, 0, s7
	s_load_dwordx4 s[0:3], s[4:5], 0x20
	s_abs_i32 s5, s16
	v_rcp_iflag_f32_e32 v1, v1
	v_and_b32_e32 v2, 0x3ff, v0
	s_xor_b32 s4, s16, s6
	v_readfirstlane_b32 s34, v2
	v_mul_f32_e32 v1, 0x4f7ffffe, v1
	v_cvt_u32_f32_e32 v1, v1
	s_lshr_b32 s17, s34, 6
	s_bfe_u32 s19, s34, 0x20006
	s_ashr_i32 s4, s4, 31
	v_readfirstlane_b32 s20, v1
	s_mul_i32 s18, s18, s20
	s_mul_hi_u32 s18, s20, s18
	s_add_i32 s20, s20, s18
	s_mul_hi_u32 s18, s5, s20
	s_mul_i32 s20, s18, s7
	s_sub_i32 s5, s5, s20
	s_add_i32 s20, s18, 1
	s_sub_i32 s21, s5, s7
	s_cmp_ge_u32 s5, s7
	s_cselect_b32 s18, s20, s18
	s_cselect_b32 s5, s21, s5
	s_add_i32 s20, s18, 1
	s_cmp_ge_u32 s5, s7
	s_cselect_b32 s5, s20, s18
	s_xor_b32 s5, s5, s4
	s_sub_i32 s4, s5, s4
	v_and_b32_e32 v1, 63, v0
	s_mul_i32 s5, s4, s6
	v_lshl_or_b32 v12, s19, 6, v1
	v_lshlrev_b32_e32 v11, 3, v2
	s_sub_i32 s5, s16, s5
	v_lshrrev_b32_e32 v3, 3, v12
	v_accvgpr_write_b32 a36, v2
	v_and_b32_e32 v2, 56, v11
	v_or_b32_e32 v4, 32, v3
	s_lshl_b32 s18, s4, 8
	s_lshl_b32 s16, s5, 8
	s_waitcnt lgkmcnt(0)
	v_mad_u64_u32 v[14:15], s[4:5], v3, s1, v[2:3]
	v_or_b32_e32 v5, 64, v3
	v_accvgpr_write_b32 a38, v14
	v_mad_u64_u32 v[14:15], s[4:5], v4, s1, v[2:3]
	v_or_b32_e32 v6, 0x60, v3
	v_accvgpr_write_b32 a40, v14
	v_mad_u64_u32 v[14:15], s[4:5], v5, s1, v[2:3]
	v_or_b32_e32 v7, 0x80, v3
	v_accvgpr_write_b32 a42, v14
	v_mad_u64_u32 v[14:15], s[4:5], v6, s1, v[2:3]
	v_or_b32_e32 v8, 0xa0, v3
	v_accvgpr_write_b32 a44, v14
	v_mad_u64_u32 v[14:15], s[4:5], v7, s1, v[2:3]
	v_or_b32_e32 v9, 0xc0, v3
	v_accvgpr_write_b32 a46, v14
	v_mad_u64_u32 v[14:15], s[4:5], v8, s1, v[2:3]
	v_or_b32_e32 v10, 0xe0, v3
	v_accvgpr_write_b32 a48, v14
	v_mad_u64_u32 v[14:15], s[4:5], v9, s1, v[2:3]
	v_accvgpr_write_b32 a50, v14
	v_mad_u64_u32 v[14:15], s[4:5], v10, s1, v[2:3]
	v_accvgpr_write_b32 a52, v14
	v_mad_u64_u32 v[14:15], s[4:5], v3, s2, v[2:3]
	v_accvgpr_write_b32 a54, v14
	v_mad_u64_u32 v[14:15], s[4:5], v4, s2, v[2:3]
	v_mad_u64_u32 v[4:5], s[4:5], v5, s2, v[2:3]
	v_accvgpr_write_b32 a58, v4
	v_mad_u64_u32 v[4:5], s[4:5], v6, s2, v[2:3]
	v_accvgpr_write_b32 a60, v4
	v_mad_u64_u32 v[4:5], s[4:5], v7, s2, v[2:3]
	v_accvgpr_write_b32 a62, v4
	v_mad_u64_u32 v[4:5], s[4:5], v8, s2, v[2:3]
	v_accvgpr_write_b32 a64, v4
	v_mad_u64_u32 v[4:5], s[4:5], v9, s2, v[2:3]
	v_mad_u64_u32 v[2:3], s[4:5], v10, s2, v[2:3]
	s_mul_i32 s4, s16, s2
	s_add_i32 s0, s0, 63
	s_ashr_i32 s5, s4, 31
	s_lshr_b32 s0, s0, 6
	s_lshl_b64 s[4:5], s[4:5], 1
	s_add_u32 s2, s10, s4
	s_mul_i32 s4, s18, s1
	v_accvgpr_write_b32 a68, v2
	s_addc_u32 s10, s11, s5
	s_ashr_i32 s5, s4, 31
	v_lshlrev_b32_e32 v2, 3, v12
	v_lshrrev_b32_e32 v3, 1, v12
	s_lshl_b64 s[4:5], s[4:5], 1
	v_bitop3_b32 v3, v3, v2, 56 bitop3:0x6c
	s_add_u32 s1, s8, s4
	v_sub_u32_e32 v2, v3, v2
	v_lshlrev_b32_e32 v3, 7, v0
	v_accvgpr_write_b32 a66, v4
	s_addc_u32 s8, s9, s5
	s_lshl_b32 s4, s19, 10
	v_and_b32_e32 v3, 0x780, v3
	v_and_b32_e32 v4, 0x70, v11
	v_and_b32_e32 v0, 48, v0
	s_add_i32 s9, s4, 0
	s_and_b32 s4, s4, 0x800
	v_bitop3_b32 v0, v3, v0, v4 bitop3:0x36
	v_ashrrev_i32_e32 v2, 3, v2
	v_or_b32_e32 v3, s4, v0
	v_bitop3_b32 v4, s4, 64, v0 bitop3:0x36
	s_lshl_b32 s4, s34, 5
	v_add_u32_e32 v2, v2, v1
	s_and_b32 s4, s4, 0x800
	v_lshlrev_b32_e32 v1, 2, v2
	v_mov_b32_e32 v224, 0
	v_or_b32_e32 v5, s4, v0
	v_bitop3_b32 v6, s4, 64, v0 bitop3:0x36
	v_add_u32_e32 v0, 0, v3
	v_lshrrev_b64 v[2:3], v2, exec
	v_add_u32_e32 v7, 0, v4
	v_add_u32_e32 v148, 0, v5
	v_and_b32_e32 v2, 1, v2
	v_mov_b32_e32 v4, v224
	v_mov_b32_e32 v5, v224
	v_accvgpr_write_b32 a39, v2
	v_mov_b32_e32 v2, v224
	v_mov_b32_e32 v3, v224
	v_accvgpr_write_b32 a77, v5
	v_accvgpr_write_b32 a81, v5
	v_mov_b64_e32 v[246:247], v[4:5]
	v_accvgpr_write_b32 a93, v5
	v_accvgpr_write_b32 a89, v5
	v_accvgpr_write_b32 a125, v5
	v_accvgpr_write_b32 a121, v5
	v_accvgpr_write_b32 a117, v5
	v_accvgpr_write_b32 a113, v5
	v_accvgpr_write_b32 a109, v5
	v_accvgpr_write_b32 a85, v5
	v_accvgpr_write_b32 a133, v5
	v_accvgpr_write_b32 a105, v5
	v_accvgpr_write_b32 a101, v5
	v_accvgpr_write_b32 a97, v5
	v_accvgpr_write_b32 a129, v5
	v_accvgpr_write_b32 a73, v5
	v_accvgpr_write_b32 a35, v5
	v_accvgpr_write_b32 a56, v14
	v_accvgpr_write_b32 a37, v12
	s_add_i32 s11, s9, 0x1000
	s_add_i32 s19, s9, 0x2000
	s_add_i32 s20, s9, 0x3000
	s_add_i32 s21, s9, 0x4000
	s_add_i32 s22, s9, 0x5000
	s_add_i32 s23, s9, 0x6000
	s_add_i32 s24, s9, 0x7000
	s_add_i32 s25, s9, 0x8000
	s_add_i32 s26, s9, 0x9000
	s_add_i32 s27, s9, 0xa000
	s_add_i32 s28, s9, 0xb000
	s_add_i32 s29, s9, 0xc000
	s_add_i32 s30, s9, 0xd000
	s_add_i32 s31, s9, 0xe000
	s_add_i32 s33, s9, 0xf000
	s_mov_b32 s7, 0x27000
	s_mov_b32 s6, 0x7ffffffe
	v_add_u32_e32 v149, 0, v6
	v_mov_b32_e32 v225, v224
	v_mov_b32_e32 v226, v224
	v_mov_b32_e32 v227, v224
	v_mov_b32_e32 v156, v224
	v_mov_b32_e32 v157, v224
	v_mov_b32_e32 v158, v224
	v_mov_b32_e32 v159, v224
	v_mov_b32_e32 v228, v224
	v_mov_b32_e32 v229, v224
	v_mov_b32_e32 v230, v224
	v_mov_b32_e32 v231, v224
	v_mov_b32_e32 v232, v224
	v_mov_b32_e32 v233, v224
	v_mov_b32_e32 v234, v224
	v_mov_b32_e32 v235, v224
	v_mov_b32_e32 v236, v224
	v_mov_b32_e32 v237, v224
	v_mov_b32_e32 v238, v224
	v_mov_b32_e32 v239, v224
	v_mov_b32_e32 v12, v224
	v_mov_b32_e32 v13, v224
	v_mov_b32_e32 v14, v224
	v_mov_b32_e32 v15, v224
	v_mov_b32_e32 v16, v224
	v_mov_b32_e32 v17, v224
	v_mov_b32_e32 v18, v224
	v_mov_b32_e32 v19, v224
	v_mov_b32_e32 v20, v224
	v_mov_b32_e32 v21, v224
	v_mov_b32_e32 v22, v224
	v_mov_b32_e32 v23, v224
	v_mov_b32_e32 v220, v224
	v_mov_b32_e32 v221, v224
	v_mov_b32_e32 v222, v224
	v_mov_b32_e32 v223, v224
	v_mov_b32_e32 v216, v224
	v_mov_b32_e32 v217, v224
	v_mov_b32_e32 v218, v224
	v_mov_b32_e32 v219, v224
	v_mov_b32_e32 v212, v224
	v_mov_b32_e32 v213, v224
	v_mov_b32_e32 v214, v224
	v_mov_b32_e32 v215, v224
	v_mov_b32_e32 v208, v224
	v_mov_b32_e32 v209, v224
	v_mov_b32_e32 v210, v224
	v_mov_b32_e32 v211, v224
	v_mov_b32_e32 v204, v224
	v_mov_b32_e32 v205, v224
	v_mov_b32_e32 v206, v224
	v_mov_b32_e32 v207, v224
	v_mov_b32_e32 v192, v224
	v_mov_b32_e32 v193, v224
	v_mov_b32_e32 v194, v224
	v_mov_b32_e32 v195, v224
	v_mov_b32_e32 v176, v224
	v_mov_b32_e32 v177, v224
	v_mov_b32_e32 v178, v224
	v_mov_b32_e32 v179, v224
	v_mov_b32_e32 v160, v224
	v_mov_b32_e32 v161, v224
	v_mov_b32_e32 v162, v224
	v_mov_b32_e32 v163, v224
	v_mov_b32_e32 v164, v224
	v_mov_b32_e32 v165, v224
	v_mov_b32_e32 v166, v224
	v_mov_b32_e32 v167, v224
	v_mov_b32_e32 v168, v224
	v_mov_b32_e32 v169, v224
	v_mov_b32_e32 v170, v224
	v_mov_b32_e32 v171, v224
	v_mov_b32_e32 v172, v224
	v_mov_b32_e32 v173, v224
	v_mov_b32_e32 v174, v224
	v_mov_b32_e32 v175, v224
	v_mov_b32_e32 v180, v224
	v_mov_b32_e32 v181, v224
	v_mov_b32_e32 v182, v224
	v_mov_b32_e32 v183, v224
	v_mov_b32_e32 v184, v224
	v_mov_b32_e32 v185, v224
	v_mov_b32_e32 v186, v224
	v_mov_b32_e32 v187, v224
	v_mov_b32_e32 v188, v224
	v_mov_b32_e32 v189, v224
	v_mov_b32_e32 v190, v224
	v_mov_b32_e32 v191, v224
	v_mov_b32_e32 v196, v224
	v_mov_b32_e32 v197, v224
	v_mov_b32_e32 v198, v224
	v_mov_b32_e32 v199, v224
	v_mov_b32_e32 v200, v224
	v_mov_b32_e32 v201, v224
	v_mov_b32_e32 v202, v224
	v_mov_b32_e32 v203, v224
	v_accvgpr_write_b32 a76, v4
	v_accvgpr_write_b32 a75, v3
	v_accvgpr_write_b32 a74, v2
	v_mov_b32_e32 v150, v224
	v_mov_b32_e32 v151, v224
	v_mov_b32_e32 v152, v224
	v_mov_b32_e32 v153, v224
	v_accvgpr_write_b32 a80, v4
	v_accvgpr_write_b32 a79, v3
	v_accvgpr_write_b32 a78, v2
	v_mov_b32_e32 v24, v224
	v_mov_b32_e32 v25, v224
	v_mov_b32_e32 v26, v224
	v_mov_b32_e32 v27, v224
	v_mov_b32_e32 v28, v224
	v_mov_b32_e32 v29, v224
	v_mov_b32_e32 v30, v224
	v_mov_b32_e32 v31, v224
	v_mov_b64_e32 v[244:245], v[2:3]
	v_accvgpr_write_b32 a92, v4
	v_accvgpr_write_b32 a91, v3
	v_accvgpr_write_b32 a90, v2
	v_mov_b32_e32 v252, v224
	v_mov_b32_e32 v253, v224
	v_mov_b32_e32 v254, v224
	v_mov_b32_e32 v255, v224
	v_mov_b32_e32 v128, v224
	v_mov_b32_e32 v129, v224
	v_mov_b32_e32 v130, v224
	v_mov_b32_e32 v131, v224
	v_mov_b32_e32 v144, v224
	v_mov_b32_e32 v145, v224
	v_mov_b32_e32 v146, v224
	v_mov_b32_e32 v147, v224
	v_mov_b32_e32 v248, v224
	v_mov_b32_e32 v249, v224
	v_mov_b32_e32 v250, v224
	v_mov_b32_e32 v251, v224
	v_mov_b32_e32 v116, v224
	v_mov_b32_e32 v117, v224
	v_mov_b32_e32 v118, v224
	v_mov_b32_e32 v119, v224
	v_mov_b32_e32 v120, v224
	v_mov_b32_e32 v121, v224
	v_mov_b32_e32 v122, v224
	v_mov_b32_e32 v123, v224
	v_mov_b32_e32 v124, v224
	v_mov_b32_e32 v125, v224
	v_mov_b32_e32 v126, v224
	v_mov_b32_e32 v127, v224
	v_mov_b32_e32 v132, v224
	v_mov_b32_e32 v133, v224
	v_mov_b32_e32 v134, v224
	v_mov_b32_e32 v135, v224
	v_mov_b32_e32 v136, v224
	v_mov_b32_e32 v137, v224
	v_mov_b32_e32 v138, v224
	v_mov_b32_e32 v139, v224
	v_accvgpr_write_b32 a88, v4
	v_accvgpr_write_b32 a87, v3
	v_accvgpr_write_b32 a86, v2
	v_accvgpr_write_b32 a124, v4
	v_accvgpr_write_b32 a123, v3
	v_accvgpr_write_b32 a122, v2
	v_accvgpr_write_b32 a120, v4
	v_accvgpr_write_b32 a119, v3
	v_accvgpr_write_b32 a118, v2
	v_accvgpr_write_b32 a116, v4
	v_accvgpr_write_b32 a115, v3
	v_accvgpr_write_b32 a114, v2
	v_accvgpr_write_b32 a112, v4
	v_accvgpr_write_b32 a111, v3
	v_accvgpr_write_b32 a110, v2
	v_accvgpr_write_b32 a108, v4
	v_accvgpr_write_b32 a107, v3
	v_accvgpr_write_b32 a106, v2
	v_accvgpr_write_b32 a84, v4
	v_accvgpr_write_b32 a83, v3
	v_accvgpr_write_b32 a82, v2
	v_mov_b32_e32 v108, v224
	v_mov_b32_e32 v109, v224
	v_mov_b32_e32 v110, v224
	v_mov_b32_e32 v111, v224
	v_mov_b32_e32 v64, v224
	v_mov_b32_e32 v65, v224
	v_mov_b32_e32 v66, v224
	v_mov_b32_e32 v67, v224
	v_mov_b32_e32 v80, v224
	v_mov_b32_e32 v81, v224
	v_mov_b32_e32 v82, v224
	v_mov_b32_e32 v83, v224
	v_mov_b32_e32 v104, v224
	v_mov_b32_e32 v105, v224
	v_mov_b32_e32 v106, v224
	v_mov_b32_e32 v107, v224
	v_mov_b32_e32 v52, v224
	v_mov_b32_e32 v53, v224
	v_mov_b32_e32 v54, v224
	v_mov_b32_e32 v55, v224
	v_mov_b32_e32 v56, v224
	v_mov_b32_e32 v57, v224
	v_mov_b32_e32 v58, v224
	v_mov_b32_e32 v59, v224
	v_mov_b32_e32 v60, v224
	v_mov_b32_e32 v61, v224
	v_mov_b32_e32 v62, v224
	v_mov_b32_e32 v63, v224
	v_mov_b32_e32 v68, v224
	v_mov_b32_e32 v69, v224
	v_mov_b32_e32 v70, v224
	v_mov_b32_e32 v71, v224
	v_mov_b32_e32 v72, v224
	v_mov_b32_e32 v73, v224
	v_mov_b32_e32 v74, v224
	v_mov_b32_e32 v75, v224
	v_accvgpr_write_b32 a132, v4
	v_accvgpr_write_b32 a131, v3
	v_accvgpr_write_b32 a130, v2
	v_accvgpr_write_b32 a104, v4
	v_accvgpr_write_b32 a103, v3
	v_accvgpr_write_b32 a102, v2
	v_accvgpr_write_b32 a100, v4
	v_accvgpr_write_b32 a99, v3
	v_accvgpr_write_b32 a98, v2
	v_accvgpr_write_b32 a96, v4
	v_accvgpr_write_b32 a95, v3
	v_accvgpr_write_b32 a94, v2
	v_accvgpr_write_b32 a128, v4
	v_accvgpr_write_b32 a127, v3
	v_accvgpr_write_b32 a126, v2
	v_accvgpr_write_b32 a72, v4
	v_accvgpr_write_b32 a71, v3
	v_accvgpr_write_b32 a70, v2
	v_mov_b32_e32 v8, v224
	v_mov_b32_e32 v9, v224
	v_mov_b32_e32 v10, v224
	v_mov_b32_e32 v11, v224
	v_accvgpr_write_b32 a34, v4
	v_accvgpr_write_b32 a33, v3
	v_accvgpr_write_b32 a32, v2
	v_bfrev_b32_e32 v6, 1
	.p2align	5, , 4
.LBB0_1:                                ; =>This Inner Loop Header: Depth=1
	s_nop 0
	v_accvgpr_read_b32 v2, a39
	v_cmp_eq_u32_e32 vcc, 1, v2
	v_accvgpr_read_b32 v2, a38
	ds_bpermute_b32 v2, v1, v2
	s_mov_b32 m0, s9
	s_and_b32 s5, s8, 0xffff
	s_mov_b32 s4, s1
	s_waitcnt lgkmcnt(0)
	v_lshlrev_b32_e32 v2, 1, v2
	v_cndmask_b32_e32 v2, v6, v2, vcc
	s_barrier
	buffer_load_dwordx4 v2, s[4:7], 0 offen lds
	v_accvgpr_read_b32 v2, a40
	ds_bpermute_b32 v2, v1, v2
	s_mov_b32 m0, s11
	s_waitcnt lgkmcnt(0)
	v_lshlrev_b32_e32 v2, 1, v2
	v_cndmask_b32_e32 v2, v6, v2, vcc
	buffer_load_dwordx4 v2, s[4:7], 0 offen lds
	v_accvgpr_read_b32 v2, a42
	ds_bpermute_b32 v2, v1, v2
	s_mov_b32 m0, s19
	s_waitcnt lgkmcnt(0)
	v_lshlrev_b32_e32 v2, 1, v2
	v_cndmask_b32_e32 v2, v6, v2, vcc
	buffer_load_dwordx4 v2, s[4:7], 0 offen lds
	v_accvgpr_read_b32 v2, a44
	ds_bpermute_b32 v2, v1, v2
	s_mov_b32 m0, s20
	s_waitcnt lgkmcnt(0)
	v_lshlrev_b32_e32 v2, 1, v2
	v_cndmask_b32_e32 v2, v6, v2, vcc
	buffer_load_dwordx4 v2, s[4:7], 0 offen lds
	v_accvgpr_read_b32 v2, a46
	ds_bpermute_b32 v2, v1, v2
	s_mov_b32 m0, s21
	s_waitcnt lgkmcnt(0)
	v_lshlrev_b32_e32 v2, 1, v2
	v_cndmask_b32_e32 v2, v6, v2, vcc
	buffer_load_dwordx4 v2, s[4:7], 0 offen lds
	v_accvgpr_read_b32 v2, a48
	ds_bpermute_b32 v2, v1, v2
	s_mov_b32 m0, s22
	s_waitcnt lgkmcnt(0)
	v_lshlrev_b32_e32 v2, 1, v2
	v_cndmask_b32_e32 v2, v6, v2, vcc
	buffer_load_dwordx4 v2, s[4:7], 0 offen lds
	v_accvgpr_read_b32 v2, a50
	ds_bpermute_b32 v2, v1, v2
	s_mov_b32 m0, s23
	s_waitcnt lgkmcnt(0)
	v_lshlrev_b32_e32 v2, 1, v2
	v_cndmask_b32_e32 v2, v6, v2, vcc
	buffer_load_dwordx4 v2, s[4:7], 0 offen lds
	v_accvgpr_read_b32 v2, a52
	ds_bpermute_b32 v2, v1, v2
	s_mov_b32 m0, s24
	s_waitcnt lgkmcnt(0)
	v_lshlrev_b32_e32 v2, 1, v2
	v_cndmask_b32_e32 v2, v6, v2, vcc
	buffer_load_dwordx4 v2, s[4:7], 0 offen lds
	v_accvgpr_read_b32 v2, a54
	ds_bpermute_b32 v2, v1, v2
	s_mov_b32 m0, s25
	s_and_b32 s5, s10, 0xffff
	s_mov_b32 s4, s2
	s_add_u32 s1, s1, 0x80
	s_waitcnt lgkmcnt(0)
	v_lshlrev_b32_e32 v2, 1, v2
	v_cndmask_b32_e32 v2, v6, v2, vcc
	buffer_load_dwordx4 v2, s[4:7], 0 offen lds
	v_accvgpr_read_b32 v2, a56
	ds_bpermute_b32 v2, v1, v2
	s_mov_b32 m0, s26
	s_addc_u32 s8, s8, 0
	s_add_u32 s2, s2, 0x80
	s_addc_u32 s10, s10, 0
	s_waitcnt lgkmcnt(0)
	v_lshlrev_b32_e32 v2, 1, v2
	v_cndmask_b32_e32 v2, v6, v2, vcc
	buffer_load_dwordx4 v2, s[4:7], 0 offen lds
	v_accvgpr_read_b32 v2, a58
	ds_bpermute_b32 v2, v1, v2
	s_mov_b32 m0, s27
	s_add_i32 s0, s0, -1
	s_cmp_lg_u32 s0, 0
	s_waitcnt lgkmcnt(0)
	v_lshlrev_b32_e32 v2, 1, v2
	v_cndmask_b32_e32 v2, v6, v2, vcc
	buffer_load_dwordx4 v2, s[4:7], 0 offen lds
	v_accvgpr_read_b32 v2, a60
	ds_bpermute_b32 v2, v1, v2
	s_mov_b32 m0, s28
	s_waitcnt lgkmcnt(0)
	v_lshlrev_b32_e32 v2, 1, v2
	v_cndmask_b32_e32 v2, v6, v2, vcc
	buffer_load_dwordx4 v2, s[4:7], 0 offen lds
	v_accvgpr_read_b32 v2, a62
	ds_bpermute_b32 v2, v1, v2
	s_mov_b32 m0, s29
	s_waitcnt lgkmcnt(0)
	v_lshlrev_b32_e32 v2, 1, v2
	v_cndmask_b32_e32 v2, v6, v2, vcc
	buffer_load_dwordx4 v2, s[4:7], 0 offen lds
	v_accvgpr_read_b32 v2, a64
	ds_bpermute_b32 v2, v1, v2
	s_mov_b32 m0, s30
	s_waitcnt lgkmcnt(0)
	v_lshlrev_b32_e32 v2, 1, v2
	v_cndmask_b32_e32 v2, v6, v2, vcc
	buffer_load_dwordx4 v2, s[4:7], 0 offen lds
	v_accvgpr_read_b32 v2, a66
	ds_bpermute_b32 v2, v1, v2
	s_mov_b32 m0, s31
	s_waitcnt lgkmcnt(0)
	v_lshlrev_b32_e32 v2, 1, v2
	v_cndmask_b32_e32 v2, v6, v2, vcc
	buffer_load_dwordx4 v2, s[4:7], 0 offen lds
	v_accvgpr_read_b32 v2, a68
	ds_bpermute_b32 v2, v1, v2
	s_mov_b32 m0, s33
	s_waitcnt lgkmcnt(0)
	v_lshlrev_b32_e32 v2, 1, v2
	v_cndmask_b32_e32 v2, v6, v2, vcc
	buffer_load_dwordx4 v2, s[4:7], 0 offen lds
	; asyncmark
	; wait_asyncmark(0)
	s_waitcnt vmcnt(0)
	s_barrier
	ds_read_b128 a[28:31], v148 offset:36864
	ds_read_b128 a[24:27], v148 offset:40960
	ds_read_b128 a[16:19], v148 offset:45056
	ds_read_b128 a[12:15], v148 offset:49152
	ds_read_b128 a[8:11], v148 offset:53248
	ds_read_b128 a[0:3], v148 offset:57344
	ds_read_b128 a[4:7], v148 offset:61440
	ds_read_b128 a[20:23], v148 offset:32768
	ds_read_b128 v[2:5], v0
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_16x16x32_f16 v[240:243], a[20:23], v[2:5], v[224:227]
	v_mfma_f32_16x16x32_f16 v[156:159], a[28:31], v[2:5], v[156:159]
	v_mfma_f32_16x16x32_f16 v[228:231], a[24:27], v[2:5], v[228:231]
	v_mfma_f32_16x16x32_f16 v[232:235], a[16:19], v[2:5], v[232:235]
	v_mfma_f32_16x16x32_f16 v[236:239], a[12:15], v[2:5], v[236:239]
	v_mfma_f32_16x16x32_f16 v[12:15], a[8:11], v[2:5], v[12:15]
	v_mfma_f32_16x16x32_f16 v[16:19], a[0:3], v[2:5], v[16:19]
	v_mfma_f32_16x16x32_f16 v[20:23], a[4:7], v[2:5], v[20:23]
	ds_read_b128 v[2:5], v0 offset:4096
	s_nop 4
	v_accvgpr_write_b32 a137, v15
	v_accvgpr_write_b32 a136, v14
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_16x16x32_f16 v[220:223], a[20:23], v[2:5], v[220:223]
	v_accvgpr_write_b32 a135, v13
	v_accvgpr_write_b32 a134, v12
	v_mfma_f32_16x16x32_f16 v[216:219], a[28:31], v[2:5], v[216:219]
	v_mfma_f32_16x16x32_f16 v[212:215], a[24:27], v[2:5], v[212:215]
	v_mfma_f32_16x16x32_f16 v[208:211], a[16:19], v[2:5], v[208:211]
	v_mfma_f32_16x16x32_f16 v[204:207], a[12:15], v[2:5], v[204:207]
	v_mfma_f32_16x16x32_f16 v[192:195], a[8:11], v[2:5], v[192:195]
	v_mfma_f32_16x16x32_f16 v[176:179], a[0:3], v[2:5], v[176:179]
	v_mfma_f32_16x16x32_f16 v[2:5], a[4:7], v[2:5], v[160:163]
	s_nop 2
	ds_read_b128 v[160:163], v0 offset:28672
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_16x16x32_f16 a[154:157], a[4:7], v[160:163], a[32:35]
	s_nop 1
	v_accvgpr_write_b32 a141, v5
	v_accvgpr_write_b32 a140, v4
	v_accvgpr_write_b32 a139, v3
	v_accvgpr_write_b32 a138, v2
	ds_read_b128 v[2:5], v0 offset:8192
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_16x16x32_f16 v[164:167], a[20:23], v[2:5], v[164:167]
	ds_read_b128 a[32:35], v149 offset:61440
	v_mfma_f32_16x16x32_f16 v[168:171], a[28:31], v[2:5], v[168:171]
	v_mfma_f32_16x16x32_f16 v[172:175], a[24:27], v[2:5], v[172:175]
	v_mfma_f32_16x16x32_f16 v[180:183], a[16:19], v[2:5], v[180:183]
	v_mfma_f32_16x16x32_f16 v[184:187], a[12:15], v[2:5], v[184:187]
	v_mfma_f32_16x16x32_f16 v[188:191], a[8:11], v[2:5], v[188:191]
	v_mfma_f32_16x16x32_f16 v[196:199], a[0:3], v[2:5], v[196:199]
	v_mfma_f32_16x16x32_f16 v[200:203], a[4:7], v[2:5], v[200:203]
	ds_read_b128 v[2:5], v0 offset:12288
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_16x16x32_f16 v[12:15], a[28:31], v[2:5], v[150:153]
	v_mfma_f32_16x16x32_f16 a[74:77], a[20:23], v[2:5], a[74:77]
	v_mfma_f32_16x16x32_f16 a[78:81], a[24:27], v[2:5], a[78:81]
	s_nop 5
	v_accvgpr_write_b32 a145, v15
	v_accvgpr_write_b32 a144, v14
	v_accvgpr_write_b32 a143, v13
	v_accvgpr_write_b32 a142, v12
	v_mfma_f32_16x16x32_f16 v[12:15], a[16:19], v[2:5], v[24:27]
	v_mfma_f32_16x16x32_f16 v[244:247], a[8:11], v[2:5], v[244:247]
	v_mfma_f32_16x16x32_f16 a[90:93], a[0:3], v[2:5], a[90:93]
	s_nop 5
	v_accvgpr_write_b32 a149, v15
	v_accvgpr_write_b32 a148, v14
	v_accvgpr_write_b32 a147, v13
	v_accvgpr_write_b32 a146, v12
	v_mfma_f32_16x16x32_f16 v[12:15], a[12:15], v[2:5], v[28:31]
	v_mfma_f32_16x16x32_f16 v[252:255], a[4:7], v[2:5], v[252:255]
	ds_read_b128 v[2:5], v0 offset:16384
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_16x16x32_f16 v[128:131], a[20:23], v[2:5], v[128:131]
	v_mfma_f32_16x16x32_f16 v[144:147], a[28:31], v[2:5], v[144:147]
	v_mfma_f32_16x16x32_f16 v[248:251], a[24:27], v[2:5], v[248:251]
	v_mfma_f32_16x16x32_f16 v[116:119], a[16:19], v[2:5], v[116:119]
	v_mfma_f32_16x16x32_f16 v[120:123], a[12:15], v[2:5], v[120:123]
	v_mfma_f32_16x16x32_f16 v[124:127], a[8:11], v[2:5], v[124:127]
	v_mfma_f32_16x16x32_f16 v[132:135], a[0:3], v[2:5], v[132:135]
	v_mfma_f32_16x16x32_f16 v[136:139], a[4:7], v[2:5], v[136:139]
	ds_read_b128 v[2:5], v0 offset:20480
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_16x16x32_f16 a[86:89], a[20:23], v[2:5], a[86:89]
	v_mfma_f32_16x16x32_f16 a[122:125], a[28:31], v[2:5], a[122:125]
	v_mfma_f32_16x16x32_f16 a[118:121], a[24:27], v[2:5], a[118:121]
	v_mfma_f32_16x16x32_f16 a[114:117], a[16:19], v[2:5], a[114:117]
	v_mfma_f32_16x16x32_f16 a[110:113], a[12:15], v[2:5], a[110:113]
	v_mfma_f32_16x16x32_f16 a[106:109], a[8:11], v[2:5], a[106:109]
	v_mfma_f32_16x16x32_f16 a[82:85], a[0:3], v[2:5], a[82:85]
	v_mfma_f32_16x16x32_f16 v[108:111], a[4:7], v[2:5], v[108:111]
	ds_read_b128 v[2:5], v0 offset:24576
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_16x16x32_f16 v[104:107], a[24:27], v[2:5], v[104:107]
	v_mfma_f32_16x16x32_f16 a[98:101], a[24:27], v[160:163], a[98:101]
	ds_read_b128 a[24:27], v149 offset:53248
	v_mfma_f32_16x16x32_f16 v[64:67], a[20:23], v[2:5], v[64:67]
	v_mfma_f32_16x16x32_f16 v[80:83], a[28:31], v[2:5], v[80:83]
	v_mfma_f32_16x16x32_f16 v[52:55], a[16:19], v[2:5], v[52:55]
	v_mfma_f32_16x16x32_f16 v[56:59], a[12:15], v[2:5], v[56:59]
	v_mfma_f32_16x16x32_f16 v[60:63], a[8:11], v[2:5], v[60:63]
	v_mfma_f32_16x16x32_f16 v[68:71], a[0:3], v[2:5], v[68:71]
	v_mfma_f32_16x16x32_f16 v[72:75], a[4:7], v[2:5], v[72:75]
	v_mov_b64_e32 v[2:3], v[8:9]
	v_mov_b64_e32 v[4:5], v[10:11]
	ds_read_b128 a[4:7], v7
	v_mfma_f32_16x16x32_f16 a[102:105], a[28:31], v[160:163], a[102:105]
	v_accvgpr_mov_b32 a28, a134
	v_accvgpr_mov_b32 a29, a135
	v_accvgpr_mov_b32 a30, a136
	v_mfma_f32_16x16x32_f16 v[2:5], a[0:3], v[160:163], v[2:5]
	ds_read_b128 a[0:3], v149 offset:32768
	v_accvgpr_mov_b32 a31, a137
	v_mfma_f32_16x16x32_f16 a[130:133], a[20:23], v[160:163], a[130:133]
	ds_read_b128 a[20:23], v149 offset:49152
	v_mfma_f32_16x16x32_f16 a[94:97], a[16:19], v[160:163], a[94:97]
	ds_read_b128 a[16:19], v149 offset:45056
	v_mfma_f32_16x16x32_f16 a[126:129], a[12:15], v[160:163], a[126:129]
	ds_read_b128 a[12:15], v149 offset:40960
	v_mfma_f32_16x16x32_f16 a[70:73], a[8:11], v[160:163], a[70:73]
	ds_read_b128 a[8:11], v149 offset:36864
	s_waitcnt lgkmcnt(5)
	v_mfma_f32_16x16x32_f16 a[134:137], a[24:27], a[4:7], a[28:31]
	s_nop 2
	ds_read_b128 a[28:31], v149 offset:57344
	s_waitcnt lgkmcnt(5)
	v_mfma_f32_16x16x32_f16 v[224:227], a[0:3], a[4:7], v[240:243]
	s_waitcnt lgkmcnt(1)
	v_mfma_f32_16x16x32_f16 v[156:159], a[8:11], a[4:7], v[156:159]
	v_mfma_f32_16x16x32_f16 v[228:231], a[12:15], a[4:7], v[228:231]
	v_mfma_f32_16x16x32_f16 v[232:235], a[16:19], a[4:7], v[232:235]
	v_mfma_f32_16x16x32_f16 v[236:239], a[20:23], a[4:7], v[236:239]
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_16x16x32_f16 v[16:19], a[28:31], a[4:7], v[16:19]
	v_mfma_f32_16x16x32_f16 v[20:23], a[32:35], a[4:7], v[20:23]
	ds_read_b128 a[4:7], v7 offset:4096
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_16x16x32_f16 v[220:223], a[0:3], a[4:7], v[220:223]
	v_mfma_f32_16x16x32_f16 v[216:219], a[8:11], a[4:7], v[216:219]
	v_mfma_f32_16x16x32_f16 v[212:215], a[12:15], a[4:7], v[212:215]
	v_mfma_f32_16x16x32_f16 v[208:211], a[16:19], a[4:7], v[208:211]
	v_mfma_f32_16x16x32_f16 v[204:207], a[20:23], a[4:7], v[204:207]
	v_mfma_f32_16x16x32_f16 v[192:195], a[24:27], a[4:7], v[192:195]
	v_mfma_f32_16x16x32_f16 v[176:179], a[28:31], a[4:7], v[176:179]
	v_mfma_f32_16x16x32_f16 a[138:141], a[32:35], a[4:7], a[138:141]
	ds_read_b128 a[4:7], v7 offset:8192
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_16x16x32_f16 v[164:167], a[0:3], a[4:7], v[164:167]
	s_nop 4
	v_accvgpr_read_b32 v163, a141
	v_mfma_f32_16x16x32_f16 v[168:171], a[8:11], a[4:7], v[168:171]
	v_accvgpr_read_b32 v162, a140
	v_accvgpr_read_b32 v161, a139
	v_accvgpr_read_b32 v160, a138
	v_mfma_f32_16x16x32_f16 v[172:175], a[12:15], a[4:7], v[172:175]
	v_mfma_f32_16x16x32_f16 v[180:183], a[16:19], a[4:7], v[180:183]
	v_mfma_f32_16x16x32_f16 v[184:187], a[20:23], a[4:7], v[184:187]
	v_mfma_f32_16x16x32_f16 v[188:191], a[24:27], a[4:7], v[188:191]
	v_mfma_f32_16x16x32_f16 v[196:199], a[28:31], a[4:7], v[196:199]
	v_mfma_f32_16x16x32_f16 v[200:203], a[32:35], a[4:7], v[200:203]
	ds_read_b128 a[4:7], v7 offset:12288
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_16x16x32_f16 a[142:145], a[8:11], a[4:7], a[142:145]
	v_mfma_f32_16x16x32_f16 a[74:77], a[0:3], a[4:7], a[74:77]
	v_mfma_f32_16x16x32_f16 a[78:81], a[12:15], a[4:7], a[78:81]
	s_nop 5
	v_accvgpr_read_b32 v153, a145
	v_accvgpr_read_b32 v152, a144
	v_accvgpr_read_b32 v151, a143
	v_accvgpr_read_b32 v150, a142
	v_accvgpr_mov_b32 a142, a146
	v_accvgpr_mov_b32 a143, a147
	v_accvgpr_mov_b32 a144, a148
	v_accvgpr_mov_b32 a145, a149
	v_accvgpr_write_b32 a149, v15
	v_accvgpr_write_b32 a148, v14
	v_accvgpr_write_b32 a147, v13
	v_accvgpr_write_b32 a146, v12
	v_mfma_f32_16x16x32_f16 a[142:145], a[16:19], a[4:7], a[142:145]
	v_accvgpr_read_b32 v12, a134
	v_accvgpr_read_b32 v13, a135
	v_accvgpr_read_b32 v14, a136
	v_mfma_f32_16x16x32_f16 a[146:149], a[20:23], a[4:7], a[146:149]
	v_accvgpr_read_b32 v15, a137
	v_mfma_f32_16x16x32_f16 v[244:247], a[24:27], a[4:7], v[244:247]
	s_nop 1
	v_accvgpr_read_b32 v24, a142
	v_accvgpr_read_b32 v25, a143
	v_accvgpr_read_b32 v26, a144
	v_mfma_f32_16x16x32_f16 a[90:93], a[28:31], a[4:7], a[90:93]
	v_accvgpr_read_b32 v28, a146
	v_accvgpr_read_b32 v29, a147
	v_accvgpr_read_b32 v30, a148
	v_mfma_f32_16x16x32_f16 v[252:255], a[32:35], a[4:7], v[252:255]
	ds_read_b128 a[4:7], v7 offset:16384
	v_accvgpr_read_b32 v31, a149
	v_accvgpr_read_b32 v27, a145
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_16x16x32_f16 v[128:131], a[0:3], a[4:7], v[128:131]
	v_mfma_f32_16x16x32_f16 v[144:147], a[8:11], a[4:7], v[144:147]
	v_mfma_f32_16x16x32_f16 v[248:251], a[12:15], a[4:7], v[248:251]
	v_mfma_f32_16x16x32_f16 v[116:119], a[16:19], a[4:7], v[116:119]
	v_mfma_f32_16x16x32_f16 v[120:123], a[20:23], a[4:7], v[120:123]
	v_mfma_f32_16x16x32_f16 v[124:127], a[24:27], a[4:7], v[124:127]
	v_mfma_f32_16x16x32_f16 v[132:135], a[28:31], a[4:7], v[132:135]
	v_mfma_f32_16x16x32_f16 v[136:139], a[32:35], a[4:7], v[136:139]
	ds_read_b128 a[4:7], v7 offset:20480
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_16x16x32_f16 a[86:89], a[0:3], a[4:7], a[86:89]
	v_mfma_f32_16x16x32_f16 a[122:125], a[8:11], a[4:7], a[122:125]
	v_mfma_f32_16x16x32_f16 a[118:121], a[12:15], a[4:7], a[118:121]
	v_mfma_f32_16x16x32_f16 a[114:117], a[16:19], a[4:7], a[114:117]
	v_mfma_f32_16x16x32_f16 a[110:113], a[20:23], a[4:7], a[110:113]
	v_mfma_f32_16x16x32_f16 a[106:109], a[24:27], a[4:7], a[106:109]
	v_mfma_f32_16x16x32_f16 a[82:85], a[28:31], a[4:7], a[82:85]
	v_mfma_f32_16x16x32_f16 v[108:111], a[32:35], a[4:7], v[108:111]
	ds_read_b128 a[4:7], v7 offset:24576
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_16x16x32_f16 v[64:67], a[0:3], a[4:7], v[64:67]
	v_mfma_f32_16x16x32_f16 v[80:83], a[8:11], a[4:7], v[80:83]
	v_mfma_f32_16x16x32_f16 v[104:107], a[12:15], a[4:7], v[104:107]
	v_mfma_f32_16x16x32_f16 v[52:55], a[16:19], a[4:7], v[52:55]
	v_mfma_f32_16x16x32_f16 v[56:59], a[20:23], a[4:7], v[56:59]
	v_mfma_f32_16x16x32_f16 v[60:63], a[24:27], a[4:7], v[60:63]
	v_mfma_f32_16x16x32_f16 v[68:71], a[28:31], a[4:7], v[68:71]
	v_mfma_f32_16x16x32_f16 v[72:75], a[32:35], a[4:7], v[72:75]
	ds_read_b128 a[4:7], v7 offset:28672
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_16x16x32_f16 a[130:133], a[0:3], a[4:7], a[130:133]
	v_mfma_f32_16x16x32_f16 a[102:105], a[8:11], a[4:7], a[102:105]
	v_mfma_f32_16x16x32_f16 a[98:101], a[12:15], a[4:7], a[98:101]
	v_mfma_f32_16x16x32_f16 a[94:97], a[16:19], a[4:7], a[94:97]
	v_mfma_f32_16x16x32_f16 a[126:129], a[20:23], a[4:7], a[126:129]
	v_mfma_f32_16x16x32_f16 a[70:73], a[24:27], a[4:7], a[70:73]
	v_mfma_f32_16x16x32_f16 v[8:11], a[28:31], a[4:7], v[2:5]
	v_mfma_f32_16x16x32_f16 a[32:35], a[32:35], a[4:7], a[154:157]
	s_cbranch_scc1 .LBB0_1
; %bb.2:                                ; %._crit_edge
	v_mov_b64_e32 v[0:1], v[232:233]
	v_mov_b64_e32 v[2:3], v[234:235]
	v_cvt_pk_f16_f32 v234, v0, v1
	v_cvt_pk_f16_f32 v235, v2, v3
	v_accvgpr_read_b32 v0, a74
	v_accvgpr_read_b32 v1, a75
	v_accvgpr_read_b32 v2, a76
	v_accvgpr_read_b32 v3, a77
	v_cvt_pk_f16_f32 v232, v236, v237
	v_cvt_pk_f16_f32 v236, v220, v221
	v_cvt_pk_f16_f32 v220, v216, v217
	v_cvt_pk_f16_f32 v216, v212, v213
	v_cvt_pk_f16_f32 v212, v208, v209
	v_cvt_pk_f16_f32 v208, v204, v205
	v_cvt_pk_f16_f32 v204, v192, v193
	v_cvt_pk_f16_f32 v192, v160, v161
	v_cvt_pk_f16_f32 v160, v0, v1
	v_cvt_pk_f16_f32 v161, v2, v3
	v_accvgpr_read_b32 v0, a78
	v_accvgpr_read_b32 v1, a79
	v_accvgpr_read_b32 v2, a80
	v_accvgpr_read_b32 v3, a81
	v_cvt_pk_f16_f32 v224, v224, v225
	v_cvt_pk_f16_f32 v225, v226, v227
	v_cvt_pk_f16_f32 v226, v156, v157
	v_cvt_pk_f16_f32 v157, v152, v153
	v_cvt_pk_f16_f32 v152, v0, v1
	v_cvt_pk_f16_f32 v153, v2, v3
	v_mov_b64_e32 v[0:1], v[244:245]
	v_mov_b64_e32 v[2:3], v[246:247]
	v_cvt_pk_f16_f32 v140, v0, v1
	v_cvt_pk_f16_f32 v141, v2, v3
	v_accvgpr_read_b32 v0, a90
	v_accvgpr_read_b32 v1, a91
	v_accvgpr_read_b32 v2, a92
	v_accvgpr_read_b32 v3, a93
	v_mov_b64_e32 v[4:5], v[128:129]
	v_mov_b64_e32 v[6:7], v[130:131]
	v_cvt_pk_f16_f32 v130, v0, v1
	v_cvt_pk_f16_f32 v131, v2, v3
	v_accvgpr_read_b32 v0, a86
	v_accvgpr_read_b32 v1, a87
	v_accvgpr_read_b32 v2, a88
	v_accvgpr_read_b32 v3, a89
	v_cvt_pk_f16_f32 v96, v0, v1
	v_cvt_pk_f16_f32 v97, v2, v3
	v_accvgpr_read_b32 v0, a114
	v_cvt_pk_f16_f32 v114, v4, v5
	v_cvt_pk_f16_f32 v115, v6, v7
	v_accvgpr_read_b32 v4, a118
	v_accvgpr_read_b32 v1, a115
	v_accvgpr_read_b32 v2, a116
	v_accvgpr_read_b32 v3, a117
	v_accvgpr_read_b32 v5, a119
	v_accvgpr_read_b32 v6, a120
	v_accvgpr_read_b32 v7, a121
	v_cvt_pk_f16_f32 v84, v0, v1
	v_cvt_pk_f16_f32 v85, v2, v3
	v_accvgpr_read_b32 v0, a106
	v_cvt_pk_f16_f32 v88, v4, v5
	v_cvt_pk_f16_f32 v89, v6, v7
	v_accvgpr_read_b32 v4, a110
	v_accvgpr_read_b32 v1, a107
	v_accvgpr_read_b32 v2, a108
	v_accvgpr_read_b32 v3, a109
	v_cvt_pk_f16_f32 v228, v228, v229
	v_cvt_pk_f16_f32 v229, v230, v231
	v_cvt_pk_f16_f32 v230, v12, v13
	v_cvt_pk_f16_f32 v231, v14, v15
	v_mov_b64_e32 v[12:13], v[144:145]
	v_accvgpr_read_b32 v5, a111
	v_accvgpr_read_b32 v6, a112
	v_accvgpr_read_b32 v7, a113
	v_cvt_pk_f16_f32 v76, v0, v1
	v_cvt_pk_f16_f32 v77, v2, v3
	v_accvgpr_read_b32 v0, a82
	v_mov_b64_e32 v[14:15], v[146:147]
	v_cvt_pk_f16_f32 v40, v4, v5
	v_cvt_pk_f16_f32 v41, v6, v7
	v_accvgpr_read_b32 v1, a83
	v_accvgpr_read_b32 v2, a84
	v_accvgpr_read_b32 v3, a85
	v_mov_b64_e32 v[4:5], v[64:65]
	v_cvt_pk_f16_f32 v112, v12, v13
	v_cvt_pk_f16_f32 v113, v14, v15
	v_accvgpr_read_b32 v12, a122
	v_mov_b64_e32 v[6:7], v[66:67]
	v_cvt_pk_f16_f32 v66, v0, v1
	v_cvt_pk_f16_f32 v67, v2, v3
	v_accvgpr_read_b32 v0, a130
	v_accvgpr_read_b32 v13, a123
	v_accvgpr_read_b32 v14, a124
	v_accvgpr_read_b32 v15, a125
	v_accvgpr_read_b32 v1, a131
	v_accvgpr_read_b32 v2, a132
	v_accvgpr_read_b32 v3, a133
	v_cvt_pk_f16_f32 v92, v12, v13
	v_cvt_pk_f16_f32 v93, v14, v15
	v_accvgpr_read_b32 v12, a102
	v_cvt_pk_f16_f32 v32, v0, v1
	v_cvt_pk_f16_f32 v33, v2, v3
	v_accvgpr_read_b32 v0, a94
	v_accvgpr_read_b32 v13, a103
	v_accvgpr_read_b32 v14, a104
	v_accvgpr_read_b32 v15, a105
	v_accvgpr_read_b32 v1, a95
	v_accvgpr_read_b32 v2, a96
	v_accvgpr_read_b32 v3, a97
	v_cvt_pk_f16_f32 v233, v238, v239
	v_cvt_pk_f16_f32 v238, v20, v21
	v_cvt_pk_f16_f32 v144, v28, v29
	v_cvt_pk_f16_f32 v20, v56, v57
	v_cvt_pk_f16_f32 v28, v12, v13
	v_cvt_pk_f16_f32 v29, v14, v15
	v_accvgpr_read_b32 v12, a126
	v_cvt_pk_f16_f32 v56, v0, v1
	v_cvt_pk_f16_f32 v57, v2, v3
	v_accvgpr_read_b32 v0, a70
	v_cvt_pk_f16_f32 v154, v16, v17
	v_cvt_pk_f16_f32 v155, v18, v19
	v_mov_b64_e32 v[16:17], v[108:109]
	v_accvgpr_read_b32 v13, a127
	v_accvgpr_read_b32 v1, a71
	v_accvgpr_read_b32 v2, a72
	v_accvgpr_read_b32 v3, a73
	v_cvt_pk_f16_f32 v64, v16, v17
	v_cvt_pk_f16_f32 v16, v12, v13
	v_cvt_pk_f16_f32 v12, v0, v1
	v_cvt_pk_f16_f32 v13, v2, v3
	v_accvgpr_read_b32 v0, a32
	v_accvgpr_read_b32 v1, a33
	v_accvgpr_read_b32 v2, a34
	v_accvgpr_read_b32 v3, a35
	v_cvt_pk_f16_f32 v0, v0, v1
	v_cvt_pk_f16_f32 v1, v2, v3
	v_accvgpr_read_b32 v2, a36
	v_and_b32_e32 v2, 15, v2
	s_lshl_b32 s0, s17, 3
	v_and_or_b32 v2, s0, 16, v2
	s_mul_i32 s0, s18, s3
	s_ashr_i32 s1, s0, 31
	v_accvgpr_read_b32 v14, a128
	v_accvgpr_read_b32 v15, a129
	s_lshl_b64 s[0:1], s[0:1], 1
	v_cvt_pk_f16_f32 v17, v14, v15
	v_accvgpr_read_b32 v14, a37
	s_add_u32 s2, s12, s0
	v_cvt_pk_f16_f32 v50, v4, v5
	v_cvt_pk_f16_f32 v51, v6, v7
	v_accvgpr_read_b32 v4, a98
	v_lshrrev_b32_e32 v14, 2, v14
	s_addc_u32 s4, s13, s1
	s_ashr_i32 s17, s16, 31
	v_mov_b64_e32 v[18:19], v[110:111]
	v_accvgpr_read_b32 v5, a99
	v_accvgpr_read_b32 v6, a100
	v_accvgpr_read_b32 v7, a101
	v_and_b32_e32 v14, 28, v14
	s_lshl_b64 s[0:1], s[16:17], 1
	v_cvt_pk_f16_f32 v239, v22, v23
	v_cvt_pk_f16_f32 v148, v24, v25
	v_cvt_pk_f16_f32 v149, v26, v27
	v_cvt_pk_f16_f32 v145, v30, v31
	v_cvt_pk_f16_f32 v65, v18, v19
	v_cvt_pk_f16_f32 v24, v4, v5
	v_cvt_pk_f16_f32 v25, v6, v7
	v_cvt_pk_f16_f32 v4, v8, v9
	v_cvt_pk_f16_f32 v5, v10, v11
	v_or_b32_e32 v3, 32, v2
	v_or_b32_e32 v6, 64, v2
	v_or_b32_e32 v7, 0x60, v2
	v_or_b32_e32 v8, 0x80, v2
	v_or_b32_e32 v9, 0xa0, v2
	v_or_b32_e32 v10, 0xc0, v2
	v_or_b32_e32 v11, 0xe0, v2
	v_or_b32_e32 v15, 32, v14
	v_or_b32_e32 v18, 64, v14
	v_or_b32_e32 v19, 0x60, v14
	v_or_b32_e32 v22, 0x80, v14
	v_or_b32_e32 v23, 0xa0, v14
	v_or_b32_e32 v26, 0xc0, v14
	v_or_b32_e32 v27, 0xe0, v14
	s_add_u32 s36, s2, s0
	v_mul_lo_u32 v30, v2, s3
	v_cmp_gt_i32_e64 s[28:29], s14, v2
	v_cmp_gt_i32_e64 s[30:31], s15, v14
	v_cvt_pk_f16_f32 v78, v52, v53
	v_cvt_pk_f16_f32 v79, v54, v55
	s_addc_u32 s33, s4, s1
	v_mul_lo_u32 v31, v3, s3
	v_mul_lo_u32 v46, v6, s3
	v_mul_lo_u32 v47, v7, s3
	v_mul_lo_u32 v52, v8, s3
	v_mul_lo_u32 v53, v9, s3
	v_mul_lo_u32 v54, v10, s3
	v_mul_lo_u32 v55, v11, s3
	v_cmp_gt_i32_e64 s[26:27], s14, v3
	v_cmp_gt_i32_e64 s[24:25], s14, v6
	v_cmp_gt_i32_e64 s[22:23], s14, v7
	v_cmp_gt_i32_e64 s[20:21], s14, v8
	v_cmp_gt_i32_e64 s[18:19], s14, v9
	v_cmp_gt_i32_e64 s[16:17], s14, v10
	v_cmp_gt_i32_e32 vcc, s14, v11
	v_cmp_gt_i32_e64 s[12:13], s15, v15
	v_cmp_gt_i32_e64 s[10:11], s15, v18
	v_cmp_gt_i32_e64 s[8:9], s15, v19
	v_cmp_gt_i32_e64 s[6:7], s15, v22
	v_cmp_gt_i32_e64 s[4:5], s15, v23
	v_cmp_gt_i32_e64 s[2:3], s15, v26
	v_cmp_gt_i32_e64 s[0:1], s15, v27
	v_add_lshl_u32 v2, v14, v30, 1
	v_bfrev_b32_e32 v3, 1
	s_and_b64 s[14:15], s[28:29], s[30:31]
	s_and_b32 s37, s33, 0xffff
	s_mov_b32 s39, 0x27000
	s_mov_b32 s38, 0x7ffffffe
	v_cndmask_b32_e64 v2, v3, v2, s[14:15]
	buffer_store_dwordx2 v[224:225], v2, s[36:39], 0 offen
	v_add_lshl_u32 v2, v15, v30, 1
	s_and_b64 s[14:15], s[28:29], s[12:13]
	v_cvt_pk_f16_f32 v227, v158, v159
	v_cndmask_b32_e64 v2, v3, v2, s[14:15]
	buffer_store_dwordx2 v[226:227], v2, s[36:39], 0 offen
	v_add_lshl_u32 v2, v18, v30, 1
	s_and_b64 s[14:15], s[28:29], s[10:11]
	v_cndmask_b32_e64 v2, v3, v2, s[14:15]
	buffer_store_dwordx2 v[228:229], v2, s[36:39], 0 offen
	v_add_lshl_u32 v2, v19, v30, 1
	s_and_b64 s[14:15], s[28:29], s[8:9]
	v_cndmask_b32_e64 v2, v3, v2, s[14:15]
	buffer_store_dwordx2 v[234:235], v2, s[36:39], 0 offen
	v_add_lshl_u32 v2, v22, v30, 1
	s_and_b64 s[14:15], s[28:29], s[6:7]
	v_cndmask_b32_e64 v2, v3, v2, s[14:15]
	buffer_store_dwordx2 v[232:233], v2, s[36:39], 0 offen
	v_add_lshl_u32 v2, v23, v30, 1
	s_and_b64 s[14:15], s[28:29], s[4:5]
	v_cndmask_b32_e64 v2, v3, v2, s[14:15]
	buffer_store_dwordx2 v[230:231], v2, s[36:39], 0 offen
	v_add_lshl_u32 v2, v26, v30, 1
	s_and_b64 s[14:15], s[28:29], s[2:3]
	v_cndmask_b32_e64 v2, v3, v2, s[14:15]
	buffer_store_dwordx2 v[154:155], v2, s[36:39], 0 offen
	v_add_lshl_u32 v2, v27, v30, 1
	s_and_b64 s[14:15], s[28:29], s[0:1]
	v_cndmask_b32_e64 v2, v3, v2, s[14:15]
	buffer_store_dwordx2 v[238:239], v2, s[36:39], 0 offen
	v_add_lshl_u32 v2, v31, v14, 1
	s_and_b64 s[14:15], s[26:27], s[30:31]
	v_cvt_pk_f16_f32 v237, v222, v223
	v_cndmask_b32_e64 v2, v3, v2, s[14:15]
	buffer_store_dwordx2 v[236:237], v2, s[36:39], 0 offen
	v_add_lshl_u32 v2, v15, v31, 1
	s_and_b64 s[14:15], s[26:27], s[12:13]
	v_cvt_pk_f16_f32 v221, v218, v219
	v_cndmask_b32_e64 v2, v3, v2, s[14:15]
	buffer_store_dwordx2 v[220:221], v2, s[36:39], 0 offen
	v_add_lshl_u32 v2, v18, v31, 1
	s_and_b64 s[14:15], s[26:27], s[10:11]
	v_cvt_pk_f16_f32 v217, v214, v215
	v_cndmask_b32_e64 v2, v3, v2, s[14:15]
	buffer_store_dwordx2 v[216:217], v2, s[36:39], 0 offen
	v_add_lshl_u32 v2, v19, v31, 1
	s_and_b64 s[14:15], s[26:27], s[8:9]
	v_cvt_pk_f16_f32 v213, v210, v211
	v_cndmask_b32_e64 v2, v3, v2, s[14:15]
	buffer_store_dwordx2 v[212:213], v2, s[36:39], 0 offen
	v_add_lshl_u32 v2, v22, v31, 1
	s_and_b64 s[14:15], s[26:27], s[6:7]
	v_cvt_pk_f16_f32 v209, v206, v207
	v_cndmask_b32_e64 v2, v3, v2, s[14:15]
	buffer_store_dwordx2 v[208:209], v2, s[36:39], 0 offen
	v_add_lshl_u32 v2, v23, v31, 1
	s_and_b64 s[14:15], s[26:27], s[4:5]
	v_cvt_pk_f16_f32 v205, v194, v195
	v_cndmask_b32_e64 v2, v3, v2, s[14:15]
	buffer_store_dwordx2 v[204:205], v2, s[36:39], 0 offen
	v_add_lshl_u32 v2, v26, v31, 1
	s_and_b64 s[14:15], s[26:27], s[2:3]
	v_cvt_pk_f16_f32 v194, v176, v177
	v_cvt_pk_f16_f32 v195, v178, v179
	v_cndmask_b32_e64 v2, v3, v2, s[14:15]
	buffer_store_dwordx2 v[194:195], v2, s[36:39], 0 offen
	v_add_lshl_u32 v2, v27, v31, 1
	s_and_b64 s[14:15], s[26:27], s[0:1]
	v_cvt_pk_f16_f32 v193, v162, v163
	v_cndmask_b32_e64 v2, v3, v2, s[14:15]
	buffer_store_dwordx2 v[192:193], v2, s[36:39], 0 offen
	v_add_lshl_u32 v2, v46, v14, 1
	s_and_b64 s[14:15], s[24:25], s[30:31]
	v_cvt_pk_f16_f32 v178, v164, v165
	v_cvt_pk_f16_f32 v179, v166, v167
	v_cndmask_b32_e64 v2, v3, v2, s[14:15]
	buffer_store_dwordx2 v[178:179], v2, s[36:39], 0 offen
	v_add_lshl_u32 v2, v15, v46, 1
	s_and_b64 s[14:15], s[24:25], s[12:13]
	v_cvt_pk_f16_f32 v176, v168, v169
	v_cvt_pk_f16_f32 v177, v170, v171
	v_cndmask_b32_e64 v2, v3, v2, s[14:15]
	buffer_store_dwordx2 v[176:177], v2, s[36:39], 0 offen
	v_add_lshl_u32 v2, v18, v46, 1
	s_and_b64 s[14:15], s[24:25], s[10:11]
	v_cvt_pk_f16_f32 v172, v172, v173
	v_cvt_pk_f16_f32 v173, v174, v175
	v_cndmask_b32_e64 v2, v3, v2, s[14:15]
	buffer_store_dwordx2 v[172:173], v2, s[36:39], 0 offen
	v_add_lshl_u32 v2, v19, v46, 1
	s_and_b64 s[14:15], s[24:25], s[8:9]
	v_cvt_pk_f16_f32 v170, v180, v181
	v_cvt_pk_f16_f32 v171, v182, v183
	v_cndmask_b32_e64 v2, v3, v2, s[14:15]
	buffer_store_dwordx2 v[170:171], v2, s[36:39], 0 offen
	v_add_lshl_u32 v2, v22, v46, 1
	s_and_b64 s[14:15], s[24:25], s[6:7]
	v_cvt_pk_f16_f32 v168, v184, v185
	v_cvt_pk_f16_f32 v169, v186, v187
	v_cndmask_b32_e64 v2, v3, v2, s[14:15]
	buffer_store_dwordx2 v[168:169], v2, s[36:39], 0 offen
	v_add_lshl_u32 v2, v23, v46, 1
	s_and_b64 s[14:15], s[24:25], s[4:5]
	v_cvt_pk_f16_f32 v166, v188, v189
	v_cvt_pk_f16_f32 v167, v190, v191
	v_cndmask_b32_e64 v2, v3, v2, s[14:15]
	buffer_store_dwordx2 v[166:167], v2, s[36:39], 0 offen
	v_add_lshl_u32 v2, v26, v46, 1
	s_and_b64 s[14:15], s[24:25], s[2:3]
	v_cvt_pk_f16_f32 v164, v196, v197
	v_cvt_pk_f16_f32 v165, v198, v199
	v_cndmask_b32_e64 v2, v3, v2, s[14:15]
	buffer_store_dwordx2 v[164:165], v2, s[36:39], 0 offen
	v_add_lshl_u32 v2, v27, v46, 1
	s_and_b64 s[14:15], s[24:25], s[0:1]
	v_cvt_pk_f16_f32 v162, v200, v201
	v_cvt_pk_f16_f32 v163, v202, v203
	v_cndmask_b32_e64 v2, v3, v2, s[14:15]
	buffer_store_dwordx2 v[162:163], v2, s[36:39], 0 offen
	v_add_lshl_u32 v2, v47, v14, 1
	s_and_b64 s[14:15], s[22:23], s[30:31]
	v_cndmask_b32_e64 v2, v3, v2, s[14:15]
	buffer_store_dwordx2 v[160:161], v2, s[36:39], 0 offen
	v_add_lshl_u32 v2, v15, v47, 1
	s_and_b64 s[14:15], s[22:23], s[12:13]
	v_cvt_pk_f16_f32 v156, v150, v151
	v_cndmask_b32_e64 v2, v3, v2, s[14:15]
	buffer_store_dwordx2 v[156:157], v2, s[36:39], 0 offen
	v_add_lshl_u32 v2, v18, v47, 1
	s_and_b64 s[14:15], s[22:23], s[10:11]
	v_cndmask_b32_e64 v2, v3, v2, s[14:15]
	buffer_store_dwordx2 v[152:153], v2, s[36:39], 0 offen
	v_add_lshl_u32 v2, v19, v47, 1
	s_and_b64 s[14:15], s[22:23], s[8:9]
	v_cndmask_b32_e64 v2, v3, v2, s[14:15]
	buffer_store_dwordx2 v[148:149], v2, s[36:39], 0 offen
	v_add_lshl_u32 v2, v22, v47, 1
	s_and_b64 s[14:15], s[22:23], s[6:7]
	v_cndmask_b32_e64 v2, v3, v2, s[14:15]
	buffer_store_dwordx2 v[144:145], v2, s[36:39], 0 offen
	v_add_lshl_u32 v2, v23, v47, 1
	s_and_b64 s[14:15], s[22:23], s[4:5]
	v_cndmask_b32_e64 v2, v3, v2, s[14:15]
	buffer_store_dwordx2 v[140:141], v2, s[36:39], 0 offen
	v_add_lshl_u32 v2, v26, v47, 1
	s_and_b64 s[14:15], s[22:23], s[2:3]
	v_cndmask_b32_e64 v2, v3, v2, s[14:15]
	buffer_store_dwordx2 v[130:131], v2, s[36:39], 0 offen
	v_add_lshl_u32 v2, v27, v47, 1
	s_and_b64 s[14:15], s[22:23], s[0:1]
	v_cvt_pk_f16_f32 v128, v252, v253
	v_cvt_pk_f16_f32 v129, v254, v255
	v_cndmask_b32_e64 v2, v3, v2, s[14:15]
	buffer_store_dwordx2 v[128:129], v2, s[36:39], 0 offen
	v_add_lshl_u32 v2, v52, v14, 1
	s_and_b64 s[14:15], s[20:21], s[30:31]
	v_cndmask_b32_e64 v2, v3, v2, s[14:15]
	buffer_store_dwordx2 v[114:115], v2, s[36:39], 0 offen
	v_add_lshl_u32 v2, v15, v52, 1
	s_and_b64 s[14:15], s[20:21], s[12:13]
	v_cndmask_b32_e64 v2, v3, v2, s[14:15]
	buffer_store_dwordx2 v[112:113], v2, s[36:39], 0 offen
	v_add_lshl_u32 v2, v18, v52, 1
	s_and_b64 s[14:15], s[20:21], s[10:11]
	v_cvt_pk_f16_f32 v108, v248, v249
	v_cvt_pk_f16_f32 v109, v250, v251
	v_cndmask_b32_e64 v2, v3, v2, s[14:15]
	buffer_store_dwordx2 v[108:109], v2, s[36:39], 0 offen
	v_add_lshl_u32 v2, v19, v52, 1
	s_and_b64 s[14:15], s[20:21], s[8:9]
	v_cvt_pk_f16_f32 v86, v116, v117
	v_cvt_pk_f16_f32 v87, v118, v119
	v_cndmask_b32_e64 v2, v3, v2, s[14:15]
	buffer_store_dwordx2 v[86:87], v2, s[36:39], 0 offen
	v_add_lshl_u32 v2, v22, v52, 1
	s_and_b64 s[14:15], s[20:21], s[6:7]
	v_cvt_pk_f16_f32 v42, v120, v121
	v_cvt_pk_f16_f32 v43, v122, v123
	v_cndmask_b32_e64 v2, v3, v2, s[14:15]
	buffer_store_dwordx2 v[42:43], v2, s[36:39], 0 offen
	v_add_lshl_u32 v2, v23, v52, 1
	s_and_b64 s[14:15], s[20:21], s[4:5]
	v_cvt_pk_f16_f32 v102, v124, v125
	v_cvt_pk_f16_f32 v103, v126, v127
	v_cndmask_b32_e64 v2, v3, v2, s[14:15]
	buffer_store_dwordx2 v[102:103], v2, s[36:39], 0 offen
	v_add_lshl_u32 v2, v26, v52, 1
	s_and_b64 s[14:15], s[20:21], s[2:3]
	v_cvt_pk_f16_f32 v100, v132, v133
	v_cvt_pk_f16_f32 v101, v134, v135
	v_cndmask_b32_e64 v2, v3, v2, s[14:15]
	buffer_store_dwordx2 v[100:101], v2, s[36:39], 0 offen
	v_add_lshl_u32 v2, v27, v52, 1
	s_and_b64 s[14:15], s[20:21], s[0:1]
	v_cvt_pk_f16_f32 v98, v136, v137
	v_cvt_pk_f16_f32 v99, v138, v139
	v_cndmask_b32_e64 v2, v3, v2, s[14:15]
	buffer_store_dwordx2 v[98:99], v2, s[36:39], 0 offen
	v_add_lshl_u32 v2, v53, v14, 1
	s_and_b64 s[14:15], s[18:19], s[30:31]
	v_cndmask_b32_e64 v2, v3, v2, s[14:15]
	buffer_store_dwordx2 v[96:97], v2, s[36:39], 0 offen
	v_add_lshl_u32 v2, v15, v53, 1
	s_and_b64 s[14:15], s[18:19], s[12:13]
	v_cndmask_b32_e64 v2, v3, v2, s[14:15]
	buffer_store_dwordx2 v[92:93], v2, s[36:39], 0 offen
	v_add_lshl_u32 v2, v18, v53, 1
	s_and_b64 s[14:15], s[18:19], s[10:11]
	v_cndmask_b32_e64 v2, v3, v2, s[14:15]
	buffer_store_dwordx2 v[88:89], v2, s[36:39], 0 offen
	v_add_lshl_u32 v2, v19, v53, 1
	s_and_b64 s[14:15], s[18:19], s[8:9]
	v_cndmask_b32_e64 v2, v3, v2, s[14:15]
	buffer_store_dwordx2 v[84:85], v2, s[36:39], 0 offen
	v_add_lshl_u32 v2, v22, v53, 1
	s_and_b64 s[14:15], s[18:19], s[6:7]
	v_cndmask_b32_e64 v2, v3, v2, s[14:15]
	buffer_store_dwordx2 v[40:41], v2, s[36:39], 0 offen
	v_add_lshl_u32 v2, v23, v53, 1
	s_and_b64 s[14:15], s[18:19], s[4:5]
	v_cndmask_b32_e64 v2, v3, v2, s[14:15]
	buffer_store_dwordx2 v[76:77], v2, s[36:39], 0 offen
	v_add_lshl_u32 v2, v26, v53, 1
	s_and_b64 s[14:15], s[18:19], s[2:3]
	v_cndmask_b32_e64 v2, v3, v2, s[14:15]
	buffer_store_dwordx2 v[66:67], v2, s[36:39], 0 offen
	v_add_lshl_u32 v2, v27, v53, 1
	s_and_b64 s[14:15], s[18:19], s[0:1]
	v_cndmask_b32_e64 v2, v3, v2, s[14:15]
	buffer_store_dwordx2 v[64:65], v2, s[36:39], 0 offen
	v_add_lshl_u32 v2, v54, v14, 1
	s_and_b64 s[14:15], s[16:17], s[30:31]
	v_cndmask_b32_e64 v2, v3, v2, s[14:15]
	buffer_store_dwordx2 v[50:51], v2, s[36:39], 0 offen
	v_add_lshl_u32 v2, v15, v54, 1
	s_and_b64 s[14:15], s[16:17], s[12:13]
	v_cvt_pk_f16_f32 v48, v80, v81
	v_cvt_pk_f16_f32 v49, v82, v83
	v_cndmask_b32_e64 v2, v3, v2, s[14:15]
	buffer_store_dwordx2 v[48:49], v2, s[36:39], 0 offen
	v_add_lshl_u32 v2, v18, v54, 1
	s_and_b64 s[14:15], s[16:17], s[10:11]
	v_cvt_pk_f16_f32 v44, v104, v105
	v_cvt_pk_f16_f32 v45, v106, v107
	v_cndmask_b32_e64 v2, v3, v2, s[14:15]
	buffer_store_dwordx2 v[44:45], v2, s[36:39], 0 offen
	v_add_lshl_u32 v2, v19, v54, 1
	s_and_b64 s[14:15], s[16:17], s[8:9]
	v_cndmask_b32_e64 v2, v3, v2, s[14:15]
	buffer_store_dwordx2 v[78:79], v2, s[36:39], 0 offen
	v_add_lshl_u32 v2, v22, v54, 1
	s_and_b64 s[14:15], s[16:17], s[6:7]
	v_cvt_pk_f16_f32 v21, v58, v59
	v_cndmask_b32_e64 v2, v3, v2, s[14:15]
	buffer_store_dwordx2 v[20:21], v2, s[36:39], 0 offen
	v_add_lshl_u32 v2, v23, v54, 1
	s_and_b64 s[14:15], s[16:17], s[4:5]
	v_cvt_pk_f16_f32 v38, v60, v61
	v_cvt_pk_f16_f32 v39, v62, v63
	v_cndmask_b32_e64 v2, v3, v2, s[14:15]
	buffer_store_dwordx2 v[38:39], v2, s[36:39], 0 offen
	v_add_lshl_u32 v2, v26, v54, 1
	s_and_b64 s[14:15], s[16:17], s[2:3]
	v_cvt_pk_f16_f32 v36, v68, v69
	v_cvt_pk_f16_f32 v37, v70, v71
	v_cndmask_b32_e64 v2, v3, v2, s[14:15]
	buffer_store_dwordx2 v[36:37], v2, s[36:39], 0 offen
	v_add_lshl_u32 v2, v27, v54, 1
	s_and_b64 s[14:15], s[16:17], s[0:1]
	v_cvt_pk_f16_f32 v34, v72, v73
	v_cvt_pk_f16_f32 v35, v74, v75
	v_cndmask_b32_e64 v2, v3, v2, s[14:15]
	buffer_store_dwordx2 v[34:35], v2, s[36:39], 0 offen
	v_add_lshl_u32 v2, v55, v14, 1
	s_and_b64 s[14:15], vcc, s[30:31]
	v_cndmask_b32_e64 v2, v3, v2, s[14:15]
	buffer_store_dwordx2 v[32:33], v2, s[36:39], 0 offen
	v_add_lshl_u32 v2, v15, v55, 1
	s_and_b64 s[12:13], vcc, s[12:13]
	v_cndmask_b32_e64 v2, v3, v2, s[12:13]
	buffer_store_dwordx2 v[28:29], v2, s[36:39], 0 offen
	v_add_lshl_u32 v2, v18, v55, 1
	s_and_b64 s[10:11], vcc, s[10:11]
	v_cndmask_b32_e64 v2, v3, v2, s[10:11]
	buffer_store_dwordx2 v[24:25], v2, s[36:39], 0 offen
	v_add_lshl_u32 v2, v19, v55, 1
	s_and_b64 s[8:9], vcc, s[8:9]
	v_cndmask_b32_e64 v2, v3, v2, s[8:9]
	buffer_store_dwordx2 v[56:57], v2, s[36:39], 0 offen
	v_add_lshl_u32 v2, v22, v55, 1
	s_and_b64 s[6:7], vcc, s[6:7]
	v_cndmask_b32_e64 v2, v3, v2, s[6:7]
	buffer_store_dwordx2 v[16:17], v2, s[36:39], 0 offen
	v_add_lshl_u32 v2, v23, v55, 1
	s_and_b64 s[4:5], vcc, s[4:5]
	v_cndmask_b32_e64 v2, v3, v2, s[4:5]
	buffer_store_dwordx2 v[12:13], v2, s[36:39], 0 offen
	v_add_lshl_u32 v2, v26, v55, 1
	s_and_b64 s[2:3], vcc, s[2:3]
	v_cndmask_b32_e64 v2, v3, v2, s[2:3]
	buffer_store_dwordx2 v[4:5], v2, s[36:39], 0 offen
	v_add_lshl_u32 v2, v27, v55, 1
	s_and_b64 vcc, vcc, s[0:1]
	v_cndmask_b32_e32 v2, v3, v2, vcc
	buffer_store_dwordx2 v[0:1], v2, s[36:39], 0 offen
	s_endpgm
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel v3_lds_swizzling
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
		.amdhsa_next_free_vgpr 414
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
	.size	v3_lds_swizzling, .Lfunc_end0-v3_lds_swizzling
	.cfi_endproc
                                        ; -- End function
	.set v3_lds_swizzling.num_vgpr, 256
	.set v3_lds_swizzling.num_agpr, 158
	.set v3_lds_swizzling.numbered_sgpr, 40
	.set v3_lds_swizzling.num_named_barrier, 0
	.set v3_lds_swizzling.private_seg_size, 0
	.set v3_lds_swizzling.uses_vcc, 1
	.set v3_lds_swizzling.uses_flat_scratch, 0
	.set v3_lds_swizzling.has_dyn_sized_stack, 0
	.set v3_lds_swizzling.has_recursion, 0
	.set v3_lds_swizzling.has_indirect_call, 0
	.section	.AMDGPU.csdata,"",@progbits
; Kernel info:
; codeLenInByte = 8712
; TotalNumSgprs: 46
; NumVgprs: 256
; NumAgprs: 158
; TotalNumVgprs: 414
; ScratchSize: 0
; MemoryBound: 0
; FloatMode: 240
; IeeeMode: 1
; LDSByteSize: 0 bytes/workgroup (compile time only)
; SGPRBlocks: 5
; VGPRBlocks: 51
; NumSGPRsForWavesPerEU: 46
; NumVGPRsForWavesPerEU: 414
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
	.byte	51                              ; DW_AT_call_line
	.byte	17                              ; DW_AT_call_column
	.byte	5                               ; Abbrev [5] 0x55:0xc DW_TAG_inlined_subroutine
	.long	42                              ; DW_AT_abstract_origin
	.long	.Ldebug_ranges0                 ; DW_AT_ranges
	.byte	1                               ; DW_AT_call_file
	.byte	97                              ; DW_AT_call_line
	.byte	16                              ; DW_AT_call_column
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
	.asciz	"kernels/gemm/a16w16/v3_lds" ; string offset=24 ; kernels/gemm/a16w16/v3_lds
.Linfo_string3:
	.asciz	"v3_lds_swizzling"              ; string offset=85 ; v3_lds_swizzling
	.section	".note.GNU-stack","",@progbits
	.amdgpu_metadata
---
amdhsa.kernels:
  - .agpr_count:     158
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
    .name:           v3_lds_swizzling
    .private_segment_fixed_size: 0
    .sgpr_count:     46
    .sgpr_spill_count: 0
    .symbol:         v3_lds_swizzling.kd
    .uniform_work_group_size: 1
    .uses_dynamic_stack: false
    .vgpr_count:     414
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
