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
	v_and_b32_e32 v4, 0x3ff, v0
	v_rcp_iflag_f32_e32 v1, v1
	s_abs_i32 s5, s16
	v_readfirstlane_b32 s34, v4
	s_bfe_u32 s19, s34, 0x20006
	v_mul_f32_e32 v1, 0x4f7ffffe, v1
	v_cvt_u32_f32_e32 v1, v1
	s_xor_b32 s4, s16, s6
	s_lshr_b32 s17, s34, 6
	s_ashr_i32 s4, s4, 31
	v_readfirstlane_b32 s20, v1
	s_mul_i32 s18, s18, s20
	s_mul_hi_u32 s18, s20, s18
	s_add_i32 s20, s20, s18
	s_mul_hi_u32 s18, s5, s20
	s_mul_i32 s20, s18, s7
	v_and_b32_e32 v1, 63, v0
	s_sub_i32 s5, s5, s20
	v_lshl_or_b32 v1, s19, 6, v1
	s_add_i32 s20, s18, 1
	s_sub_i32 s21, s5, s7
	v_accvgpr_write_b32 a117, v1
	v_lshrrev_b32_e32 v1, 3, v1
	v_accvgpr_write_b32 a116, v4
	v_lshlrev_b32_e32 v4, 3, v4
	s_cmp_ge_u32 s5, s7
	v_or_b32_e32 v2, 0xe0, v1
	v_and_b32_e32 v4, 56, v4
	v_or_b32_e32 v5, 0xc0, v1
	v_or_b32_e32 v7, 0xa0, v1
	v_or_b32_e32 v9, 0x80, v1
	v_or_b32_e32 v11, 0x60, v1
	v_or_b32_e32 v13, 64, v1
	v_or_b32_e32 v15, 32, v1
	s_waitcnt lgkmcnt(0)
	v_mul_lo_u32 v17, v1, s2
	v_mul_lo_u32 v1, v1, s1
	s_cselect_b32 s18, s20, s18
	v_mul_lo_u32 v16, v15, s2
	v_mul_lo_u32 v15, v15, s1
	v_add_lshl_u32 v1, v1, v4, 1
	s_cselect_b32 s5, s21, s5
	s_add_i32 s20, s18, 1
	v_mul_lo_u32 v14, v13, s2
	v_mul_lo_u32 v13, v13, s1
	v_accvgpr_write_b32 a118, v1
	v_add_lshl_u32 v1, v15, v4, 1
	s_cmp_ge_u32 s5, s7
	v_mul_lo_u32 v12, v11, s2
	v_mul_lo_u32 v11, v11, s1
	v_accvgpr_write_b32 a119, v1
	v_add_lshl_u32 v1, v13, v4, 1
	s_cselect_b32 s5, s20, s18
	v_mul_lo_u32 v10, v9, s2
	v_mul_lo_u32 v9, v9, s1
	v_accvgpr_write_b32 a120, v1
	v_add_lshl_u32 v1, v11, v4, 1
	s_xor_b32 s5, s5, s4
	v_mul_lo_u32 v8, v7, s2
	v_mul_lo_u32 v7, v7, s1
	v_accvgpr_write_b32 a121, v1
	v_add_lshl_u32 v1, v9, v4, 1
	s_sub_i32 s4, s5, s4
	v_mul_lo_u32 v6, v5, s2
	v_mul_lo_u32 v5, v5, s1
	v_accvgpr_write_b32 a122, v1
	v_add_lshl_u32 v1, v7, v4, 1
	s_mul_i32 s5, s4, s6
	v_mul_lo_u32 v3, v2, s2
	v_mul_lo_u32 v2, v2, s1
	v_accvgpr_write_b32 a123, v1
	v_add_lshl_u32 v1, v5, v4, 1
	s_sub_i32 s5, s16, s5
	v_accvgpr_write_b32 a124, v1
	v_add_lshl_u32 v1, v2, v4, 1
	s_lshl_b32 s16, s5, 8
	v_accvgpr_write_b32 a125, v1
	v_add_lshl_u32 v1, v17, v4, 1
	s_lshl_b32 s18, s4, 8
	s_mul_i32 s4, s16, s2
	v_accvgpr_write_b32 a126, v1
	v_add_lshl_u32 v1, v16, v4, 1
	s_add_i32 s0, s0, 63
	s_ashr_i32 s5, s4, 31
	v_accvgpr_write_b32 a127, v1
	v_add_lshl_u32 v1, v14, v4, 1
	s_lshr_b32 s0, s0, 6
	s_lshl_b64 s[4:5], s[4:5], 1
	v_accvgpr_write_b32 a128, v1
	v_add_lshl_u32 v1, v12, v4, 1
	s_add_u32 s2, s10, s4
	s_mul_i32 s4, s18, s1
	v_accvgpr_write_b32 a129, v1
	v_add_lshl_u32 v1, v10, v4, 1
	s_addc_u32 s10, s11, s5
	s_ashr_i32 s5, s4, 31
	v_accvgpr_write_b32 a130, v1
	v_add_lshl_u32 v1, v8, v4, 1
	s_lshl_b64 s[4:5], s[4:5], 1
	v_accvgpr_write_b32 a131, v1
	v_add_lshl_u32 v1, v6, v4, 1
	s_add_u32 s1, s8, s4
	v_accvgpr_write_b32 a132, v1
	v_add_lshl_u32 v1, v3, v4, 1
	s_addc_u32 s8, s9, s5
	v_accvgpr_write_b32 a133, v1
	v_lshlrev_b32_e32 v1, 7, v0
	v_and_b32_e32 v0, 48, v0
	s_movk_i32 s5, 0x780
	s_lshl_b32 s4, s19, 10
	v_and_or_b32 v1, v1, s5, v0
	s_lshl_b32 s5, s34, 5
	v_mov_b32_e32 v0, 0
	s_add_i32 s9, s4, 0
	s_and_b32 s5, s5, 0x800
	v_mov_b32_e32 v101, v0
	s_add_i32 s5, s5, 0
	v_mov_b32_e32 v98, v0
	s_and_b32 s4, s4, 0x800
	s_add_i32 s4, s4, 0
	v_mov_b32_e32 v99, v0
	v_mov_b32_e32 v100, v0
	v_accvgpr_write_b32 a145, v101
	s_add_i32 s11, s9, 0x1000
	s_add_i32 s19, s9, 0x2000
	s_add_i32 s21, s9, 0x4000
	s_add_i32 s22, s9, 0x5000
	v_accvgpr_write_b32 a141, v101
	s_add_i32 s23, s9, 0x6000
	v_accvgpr_write_b32 a137, v101
	s_add_i32 s24, s9, 0x7000
	s_add_i32 s25, s9, 0x8000
	s_add_i32 s26, s9, 0x9000
	s_add_i32 s27, s9, 0xa000
	s_add_i32 s28, s9, 0xb000
	s_add_i32 s20, s9, 0x3000
	s_add_i32 s29, s9, 0xc000
	s_add_i32 s30, s9, 0xd000
	s_add_i32 s31, s9, 0xe000
	s_add_i32 s33, s9, 0xf000
	s_mov_b32 s7, 0x27000
	s_mov_b32 s6, 0x7ffffffe
	v_add_u32_e32 v4, s4, v1
	v_add_u32_e32 v5, s5, v1
	v_mov_b32_e32 v1, v0
	v_mov_b32_e32 v2, v0
	v_mov_b32_e32 v3, v0
	v_mov_b32_e32 v6, v0
	v_mov_b32_e32 v7, v0
	v_mov_b32_e32 v8, v0
	v_mov_b32_e32 v9, v0
	v_mov_b32_e32 v10, v0
	v_mov_b32_e32 v11, v0
	v_mov_b32_e32 v12, v0
	v_mov_b32_e32 v13, v0
	v_mov_b32_e32 v14, v0
	v_mov_b32_e32 v15, v0
	v_mov_b32_e32 v16, v0
	v_mov_b32_e32 v17, v0
	v_mov_b32_e32 v18, v0
	v_mov_b32_e32 v19, v0
	v_mov_b32_e32 v20, v0
	v_mov_b32_e32 v21, v0
	v_mov_b32_e32 v22, v0
	v_mov_b32_e32 v23, v0
	v_mov_b32_e32 v24, v0
	v_mov_b32_e32 v25, v0
	v_mov_b32_e32 v26, v0
	v_mov_b32_e32 v27, v0
	v_mov_b32_e32 v28, v0
	v_mov_b32_e32 v29, v0
	v_mov_b32_e32 v30, v0
	v_mov_b32_e32 v31, v0
	v_mov_b32_e32 v32, v0
	v_mov_b32_e32 v33, v0
	v_mov_b32_e32 v34, v0
	v_mov_b32_e32 v35, v0
	v_mov_b32_e32 v36, v0
	v_mov_b32_e32 v37, v0
	v_mov_b32_e32 v38, v0
	v_mov_b32_e32 v39, v0
	v_mov_b32_e32 v40, v0
	v_mov_b32_e32 v41, v0
	v_mov_b32_e32 v42, v0
	v_mov_b32_e32 v43, v0
	v_mov_b32_e32 v44, v0
	v_mov_b32_e32 v45, v0
	v_mov_b32_e32 v46, v0
	v_mov_b32_e32 v47, v0
	v_mov_b32_e32 v48, v0
	v_mov_b32_e32 v49, v0
	v_mov_b32_e32 v50, v0
	v_mov_b32_e32 v51, v0
	v_mov_b32_e32 v52, v0
	v_mov_b32_e32 v53, v0
	v_mov_b32_e32 v54, v0
	v_mov_b32_e32 v55, v0
	v_mov_b32_e32 v56, v0
	v_mov_b32_e32 v57, v0
	v_mov_b32_e32 v58, v0
	v_mov_b32_e32 v59, v0
	v_mov_b32_e32 v60, v0
	v_mov_b32_e32 v61, v0
	v_mov_b32_e32 v62, v0
	v_mov_b32_e32 v63, v0
	v_mov_b32_e32 v64, v0
	v_mov_b32_e32 v65, v0
	v_mov_b32_e32 v66, v0
	v_mov_b32_e32 v67, v0
	v_mov_b32_e32 v68, v0
	v_mov_b32_e32 v69, v0
	v_mov_b32_e32 v70, v0
	v_mov_b32_e32 v71, v0
	v_mov_b32_e32 v72, v0
	v_mov_b32_e32 v73, v0
	v_mov_b32_e32 v74, v0
	v_mov_b32_e32 v75, v0
	v_mov_b32_e32 v76, v0
	v_mov_b32_e32 v77, v0
	v_mov_b32_e32 v78, v0
	v_mov_b32_e32 v79, v0
	v_mov_b32_e32 v80, v0
	v_mov_b32_e32 v81, v0
	v_mov_b32_e32 v82, v0
	v_mov_b32_e32 v83, v0
	v_mov_b32_e32 v84, v0
	v_mov_b32_e32 v85, v0
	v_mov_b32_e32 v86, v0
	v_mov_b32_e32 v87, v0
	v_mov_b32_e32 v88, v0
	v_mov_b32_e32 v89, v0
	v_mov_b32_e32 v90, v0
	v_mov_b32_e32 v91, v0
	v_mov_b32_e32 v92, v0
	v_mov_b32_e32 v93, v0
	v_mov_b32_e32 v94, v0
	v_mov_b32_e32 v95, v0
	v_mov_b32_e32 v96, v0
	v_mov_b32_e32 v97, v0
	v_accvgpr_write_b32 a140, v100
	v_accvgpr_write_b32 a139, v99
	v_accvgpr_write_b32 a138, v98
	v_mov_b32_e32 v108, v0
	v_mov_b32_e32 v109, v0
	v_mov_b32_e32 v110, v0
	v_mov_b32_e32 v111, v0
	v_mov_b32_e32 v116, v0
	v_mov_b32_e32 v117, v0
	v_mov_b32_e32 v118, v0
	v_mov_b32_e32 v119, v0
	v_mov_b32_e32 v124, v0
	v_mov_b32_e32 v125, v0
	v_mov_b32_e32 v126, v0
	v_mov_b32_e32 v127, v0
	v_mov_b32_e32 v132, v0
	v_mov_b32_e32 v133, v0
	v_mov_b32_e32 v134, v0
	v_mov_b32_e32 v135, v0
	v_mov_b32_e32 v136, v0
	v_mov_b32_e32 v137, v0
	v_mov_b32_e32 v138, v0
	v_mov_b32_e32 v139, v0
	v_mov_b32_e32 v140, v0
	v_mov_b32_e32 v141, v0
	v_mov_b32_e32 v142, v0
	v_mov_b32_e32 v143, v0
	v_mov_b32_e32 v148, v0
	v_mov_b32_e32 v149, v0
	v_mov_b32_e32 v150, v0
	v_mov_b32_e32 v151, v0
	v_mov_b32_e32 v152, v0
	v_mov_b32_e32 v153, v0
	v_mov_b32_e32 v154, v0
	v_mov_b32_e32 v155, v0
	v_mov_b32_e32 v160, v0
	v_mov_b32_e32 v161, v0
	v_mov_b32_e32 v162, v0
	v_mov_b32_e32 v163, v0
	v_mov_b32_e32 v168, v0
	v_mov_b32_e32 v169, v0
	v_mov_b32_e32 v170, v0
	v_mov_b32_e32 v171, v0
	v_mov_b32_e32 v176, v0
	v_mov_b32_e32 v177, v0
	v_mov_b32_e32 v178, v0
	v_mov_b32_e32 v179, v0
	v_mov_b32_e32 v184, v0
	v_mov_b32_e32 v185, v0
	v_mov_b32_e32 v186, v0
	v_mov_b32_e32 v187, v0
	v_mov_b32_e32 v192, v0
	v_mov_b32_e32 v193, v0
	v_mov_b32_e32 v194, v0
	v_mov_b32_e32 v195, v0
	v_mov_b32_e32 v196, v0
	v_mov_b32_e32 v197, v0
	v_mov_b32_e32 v198, v0
	v_mov_b32_e32 v199, v0
	v_mov_b32_e32 v200, v0
	v_mov_b32_e32 v201, v0
	v_mov_b32_e32 v202, v0
	v_mov_b32_e32 v203, v0
	v_mov_b32_e32 v232, v0
	v_mov_b32_e32 v233, v0
	v_mov_b32_e32 v234, v0
	v_mov_b32_e32 v235, v0
	v_mov_b32_e32 v204, v0
	v_mov_b32_e32 v205, v0
	v_mov_b32_e32 v206, v0
	v_mov_b32_e32 v207, v0
	v_mov_b32_e32 v208, v0
	v_mov_b32_e32 v209, v0
	v_mov_b32_e32 v210, v0
	v_mov_b32_e32 v211, v0
	v_mov_b32_e32 v212, v0
	v_mov_b32_e32 v213, v0
	v_mov_b32_e32 v214, v0
	v_mov_b32_e32 v215, v0
	v_mov_b32_e32 v216, v0
	v_mov_b32_e32 v217, v0
	v_mov_b32_e32 v218, v0
	v_mov_b32_e32 v219, v0
	v_mov_b32_e32 v220, v0
	v_mov_b32_e32 v221, v0
	v_mov_b32_e32 v222, v0
	v_mov_b32_e32 v223, v0
	v_mov_b32_e32 v224, v0
	v_mov_b32_e32 v225, v0
	v_mov_b32_e32 v226, v0
	v_mov_b32_e32 v227, v0
	v_mov_b32_e32 v228, v0
	v_mov_b32_e32 v229, v0
	v_mov_b32_e32 v230, v0
	v_mov_b32_e32 v231, v0
	v_mov_b32_e32 v248, v0
	v_mov_b32_e32 v249, v0
	v_mov_b32_e32 v250, v0
	v_mov_b32_e32 v251, v0
	v_accvgpr_write_b32 a136, v100
	v_accvgpr_write_b32 a135, v99
	v_accvgpr_write_b32 a134, v98
	v_accvgpr_write_b32 a144, v100
	v_accvgpr_write_b32 a143, v99
	v_accvgpr_write_b32 a142, v98
	v_mov_b32_e32 v112, v0
	v_mov_b32_e32 v113, v0
	v_mov_b32_e32 v114, v0
	v_mov_b32_e32 v115, v0
	v_mov_b32_e32 v120, v0
	v_mov_b32_e32 v121, v0
	v_mov_b32_e32 v122, v0
	v_mov_b32_e32 v123, v0
	v_mov_b32_e32 v128, v0
	v_mov_b32_e32 v129, v0
	v_mov_b32_e32 v130, v0
	v_mov_b32_e32 v131, v0
	v_mov_b32_e32 v252, v0
	v_mov_b32_e32 v253, v0
	v_mov_b32_e32 v254, v0
	v_mov_b32_e32 v255, v0
	v_mov_b32_e32 v144, v0
	v_mov_b32_e32 v145, v0
	v_mov_b32_e32 v146, v0
	v_mov_b32_e32 v147, v0
	v_mov_b32_e32 v244, v0
	v_mov_b32_e32 v245, v0
	v_mov_b32_e32 v246, v0
	v_mov_b32_e32 v247, v0
	v_mov_b32_e32 v156, v0
	v_mov_b32_e32 v157, v0
	v_mov_b32_e32 v158, v0
	v_mov_b32_e32 v159, v0
	v_mov_b32_e32 v164, v0
	v_mov_b32_e32 v165, v0
	v_mov_b32_e32 v166, v0
	v_mov_b32_e32 v167, v0
	v_mov_b32_e32 v172, v0
	v_mov_b32_e32 v173, v0
	v_mov_b32_e32 v174, v0
	v_mov_b32_e32 v175, v0
	v_mov_b32_e32 v180, v0
	v_mov_b32_e32 v181, v0
	v_mov_b32_e32 v182, v0
	v_mov_b32_e32 v183, v0
	v_mov_b32_e32 v188, v0
	v_mov_b32_e32 v189, v0
	v_mov_b32_e32 v190, v0
	v_mov_b32_e32 v191, v0
	v_mov_b32_e32 v104, v0
	v_mov_b32_e32 v105, v0
	v_mov_b32_e32 v106, v0
	v_mov_b32_e32 v107, v0
	.p2align	5, , 4
.LBB0_1:                                ; =>This Inner Loop Header: Depth=1
	s_nop 4
	v_accvgpr_write_b32 a153, v101
	v_accvgpr_write_b32 a152, v100
	v_accvgpr_write_b32 a151, v99
	v_accvgpr_write_b32 a150, v98
	v_mov_b64_e32 v[102:103], v[84:85]
	v_mov_b64_e32 v[100:101], v[82:83]
	v_mov_b64_e32 v[84:85], v[72:73]
	v_mov_b64_e32 v[82:83], v[70:71]
	v_mov_b64_e32 v[72:73], v[60:61]
	v_mov_b64_e32 v[70:71], v[58:59]
	v_mov_b64_e32 v[60:61], v[52:53]
	v_mov_b64_e32 v[58:59], v[50:51]
	v_mov_b64_e32 v[52:53], v[44:45]
	v_mov_b64_e32 v[50:51], v[42:43]
	v_mov_b64_e32 v[44:45], v[36:37]
	v_mov_b64_e32 v[42:43], v[34:35]
	v_mov_b64_e32 v[36:37], v[28:29]
	v_mov_b64_e32 v[34:35], v[26:27]
	v_mov_b64_e32 v[28:29], v[24:25]
	v_mov_b64_e32 v[26:27], v[22:23]
	v_mov_b64_e32 v[24:25], v[20:21]
	v_mov_b64_e32 v[22:23], v[18:19]
	v_mov_b64_e32 v[20:21], v[16:17]
	v_mov_b64_e32 v[18:19], v[14:15]
	v_mov_b64_e32 v[16:17], v[12:13]
	v_mov_b64_e32 v[14:15], v[10:11]
	v_mov_b64_e32 v[12:13], v[8:9]
	s_mov_b32 m0, s9
	s_and_b32 s5, s8, 0xffff
	s_mov_b32 s4, s1
	v_mov_b64_e32 v[10:11], v[6:7]
	v_accvgpr_read_b32 v6, a118
	s_barrier
	buffer_load_dwordx4 v6, s[4:7], 0 offen lds
	s_mov_b32 m0, s11
	v_accvgpr_read_b32 v6, a119
	buffer_load_dwordx4 v6, s[4:7], 0 offen lds
	s_mov_b32 m0, s19
	v_accvgpr_read_b32 v6, a120
	buffer_load_dwordx4 v6, s[4:7], 0 offen lds
	s_mov_b32 m0, s20
	v_accvgpr_read_b32 v6, a121
	buffer_load_dwordx4 v6, s[4:7], 0 offen lds
	s_mov_b32 m0, s21
	v_accvgpr_read_b32 v6, a122
	buffer_load_dwordx4 v6, s[4:7], 0 offen lds
	s_mov_b32 m0, s22
	v_accvgpr_read_b32 v6, a123
	buffer_load_dwordx4 v6, s[4:7], 0 offen lds
	s_mov_b32 m0, s23
	v_accvgpr_read_b32 v6, a124
	buffer_load_dwordx4 v6, s[4:7], 0 offen lds
	s_mov_b32 m0, s24
	v_accvgpr_read_b32 v6, a125
	buffer_load_dwordx4 v6, s[4:7], 0 offen lds
	s_mov_b32 m0, s25
	s_and_b32 s5, s10, 0xffff
	s_mov_b32 s4, s2
	v_accvgpr_read_b32 v6, a126
	buffer_load_dwordx4 v6, s[4:7], 0 offen lds
	s_mov_b32 m0, s26
	v_accvgpr_read_b32 v6, a127
	buffer_load_dwordx4 v6, s[4:7], 0 offen lds
	s_mov_b32 m0, s27
	v_accvgpr_read_b32 v6, a128
	buffer_load_dwordx4 v6, s[4:7], 0 offen lds
	s_mov_b32 m0, s28
	v_accvgpr_read_b32 v6, a129
	buffer_load_dwordx4 v6, s[4:7], 0 offen lds
	s_mov_b32 m0, s29
	v_accvgpr_read_b32 v6, a130
	buffer_load_dwordx4 v6, s[4:7], 0 offen lds
	s_mov_b32 m0, s30
	v_accvgpr_read_b32 v6, a131
	buffer_load_dwordx4 v6, s[4:7], 0 offen lds
	s_mov_b32 m0, s31
	v_accvgpr_read_b32 v6, a132
	buffer_load_dwordx4 v6, s[4:7], 0 offen lds
	s_mov_b32 m0, s33
	v_accvgpr_read_b32 v6, a133
	buffer_load_dwordx4 v6, s[4:7], 0 offen lds
	v_accvgpr_write_b32 a154, v220
	v_accvgpr_write_b32 a155, v221
	v_accvgpr_write_b32 a156, v222
	v_accvgpr_write_b32 a157, v223
	v_mov_b64_e32 v[222:223], v[210:211]
	v_mov_b64_e32 v[220:221], v[208:209]
	v_mov_b64_e32 v[210:211], v[198:199]
	v_mov_b64_e32 v[208:209], v[196:197]
	v_mov_b64_e32 v[198:199], v[178:179]
	v_mov_b64_e32 v[196:197], v[176:177]
	v_mov_b64_e32 v[178:179], v[154:155]
	v_mov_b64_e32 v[176:177], v[152:153]
	v_mov_b64_e32 v[154:155], v[138:139]
	v_mov_b64_e32 v[152:153], v[136:137]
	v_mov_b64_e32 v[138:139], v[118:119]
	v_mov_b64_e32 v[136:137], v[116:117]
	v_mov_b64_e32 v[118:119], v[92:93]
	v_mov_b64_e32 v[238:239], v[230:231]
	v_mov_b64_e32 v[116:117], v[90:91]
	v_mov_b64_e32 v[92:93], v[80:81]
	v_mov_b64_e32 v[236:237], v[228:229]
	v_mov_b64_e32 v[230:231], v[218:219]
	v_mov_b64_e32 v[90:91], v[78:79]
	v_mov_b64_e32 v[80:81], v[68:69]
	v_mov_b64_e32 v[228:229], v[216:217]
	v_mov_b64_e32 v[218:219], v[206:207]
	v_mov_b64_e32 v[78:79], v[66:67]
	v_mov_b64_e32 v[68:69], v[56:57]
	v_mov_b64_e32 v[216:217], v[204:205]
	v_mov_b64_e32 v[206:207], v[194:195]
	v_mov_b64_e32 v[242:243], v[226:227]
	v_mov_b64_e32 v[66:67], v[54:55]
	v_mov_b64_e32 v[56:57], v[48:49]
	v_mov_b64_e32 v[204:205], v[192:193]
	v_mov_b64_e32 v[194:195], v[170:171]
	v_mov_b64_e32 v[240:241], v[224:225]
	v_mov_b64_e32 v[226:227], v[214:215]
	v_mov_b64_e32 v[54:55], v[46:47]
	v_mov_b64_e32 v[48:49], v[40:41]
	v_mov_b64_e32 v[192:193], v[168:169]
	v_mov_b64_e32 v[170:171], v[150:151]
	v_mov_b64_e32 v[224:225], v[212:213]
	v_mov_b64_e32 v[214:215], v[202:203]
	v_mov_b64_e32 v[46:47], v[38:39]
	v_mov_b64_e32 v[40:41], v[32:33]
	v_mov_b64_e32 v[168:169], v[148:149]
	v_mov_b64_e32 v[150:151], v[134:135]
	v_mov_b64_e32 v[212:213], v[200:201]
	v_mov_b64_e32 v[202:203], v[186:187]
	; asyncmark
	; wait_asyncmark(0)
	s_waitcnt vmcnt(0)
	s_barrier
	ds_read_b128 a[8:11], v5 offset:32768
	ds_read_b128 a[12:15], v5 offset:36864
	ds_read_b128 a[16:19], v5 offset:40960
	ds_read_b128 a[20:23], v5 offset:45056
	ds_read_b128 a[24:27], v5 offset:49152
	ds_read_b128 a[28:31], v5 offset:53248
	ds_read_b128 a[32:35], v5 offset:57344
	ds_read_b128 a[36:39], v5 offset:61440
	ds_read_b128 a[0:3], v5 offset:61504
	v_mov_b64_e32 v[38:39], v[30:31]
	ds_read_b128 v[30:33], v4
	v_mov_b64_e32 v[148:149], v[132:133]
	v_mov_b64_e32 v[134:135], v[110:111]
	v_mov_b64_e32 v[200:201], v[184:185]
	v_mov_b64_e32 v[186:187], v[162:163]
	v_mov_b64_e32 v[132:133], v[108:109]
	v_mov_b64_e32 v[110:111], v[88:89]
	v_mov_b64_e32 v[184:185], v[160:161]
	v_mov_b64_e32 v[162:163], v[142:143]
	v_mov_b64_e32 v[108:109], v[86:87]
	v_mov_b64_e32 v[88:89], v[76:77]
	v_mov_b64_e32 v[160:161], v[140:141]
	v_mov_b64_e32 v[142:143], v[126:127]
	v_mov_b64_e32 v[86:87], v[74:75]
	v_mov_b64_e32 v[76:77], v[64:65]
	v_mov_b64_e32 v[140:141], v[124:125]
	v_mov_b64_e32 v[126:127], v[96:97]
	v_mov_b64_e32 v[74:75], v[62:63]
	ds_read_b128 v[62:65], v4 offset:4096
	ds_read_b128 a[40:43], v4 offset:8192
	ds_read_b128 a[44:47], v4 offset:12288
	ds_read_b128 a[48:51], v4 offset:16384
	ds_read_b128 a[52:55], v4 offset:20480
	ds_read_b128 a[56:59], v4 offset:24576
	ds_read_b128 v[6:9], v4 offset:28672
	ds_read_b128 a[60:63], v5 offset:32832
	ds_read_b128 a[64:67], v5 offset:36928
	ds_read_b128 a[68:71], v5 offset:41024
	ds_read_b128 a[72:75], v5 offset:45120
	ds_read_b128 a[76:79], v5 offset:49216
	ds_read_b128 a[80:83], v5 offset:53312
	ds_read_b128 a[84:87], v5 offset:57408
	ds_read_b128 a[88:91], v4 offset:64
	ds_read_b128 a[92:95], v4 offset:4160
	ds_read_b128 a[96:99], v4 offset:8256
	ds_read_b128 a[100:103], v4 offset:12352
	ds_read_b128 a[104:107], v4 offset:16448
	ds_read_b128 a[108:111], v4 offset:20544
	ds_read_b128 a[112:115], v4 offset:24640
	ds_read_b128 a[4:7], v4 offset:28736
	v_mov_b64_e32 v[124:125], v[94:95]
	s_waitcnt lgkmcnt(14)
	v_mfma_f32_16x16x32_f16 v[94:97], a[36:39], v[6:9], v[104:107]
	s_add_u32 s1, s1, 0x80
	s_addc_u32 s8, s8, 0
	s_add_u32 s2, s2, 0x80
	v_mfma_f32_16x16x32_f16 v[156:159], a[16:19], v[6:9], v[156:159]
	s_addc_u32 s10, s10, 0
	s_add_i32 s0, s0, -1
	s_cmp_lg_u32 s0, 0
	v_mfma_f32_16x16x32_f16 v[164:167], a[20:23], v[6:9], v[164:167]
	v_accvgpr_write_b32 a149, v97
	v_accvgpr_write_b32 a148, v96
	v_accvgpr_write_b32 a147, v95
	v_mfma_f32_16x16x32_f16 v[172:175], a[24:27], v[6:9], v[172:175]
	v_accvgpr_write_b32 a146, v94
	v_mfma_f32_16x16x32_f16 v[180:183], a[28:31], v[6:9], v[180:183]
	v_mfma_f32_16x16x32_f16 v[188:191], a[32:35], v[6:9], v[188:191]
	v_mfma_f32_16x16x32_f16 v[144:147], a[8:11], v[6:9], v[144:147]
	v_mfma_f32_16x16x32_f16 v[244:247], a[12:15], v[6:9], v[244:247]
	v_mfma_f32_16x16x32_f16 v[6:9], a[12:15], v[30:33], v[10:13]
	v_mfma_f32_16x16x32_f16 v[10:13], a[16:19], v[30:33], v[14:17]
	v_mfma_f32_16x16x32_f16 v[14:17], a[20:23], v[30:33], v[18:21]
	v_mfma_f32_16x16x32_f16 v[18:21], a[24:27], v[30:33], v[22:25]
	v_mfma_f32_16x16x32_f16 v[22:25], a[28:31], v[30:33], v[26:29]
	v_mfma_f32_16x16x32_f16 v[26:29], a[32:35], v[30:33], v[34:37]
	v_mfma_f32_16x16x32_f16 v[34:37], a[8:11], v[62:65], v[42:45]
	v_mfma_f32_16x16x32_f16 v[42:45], a[16:19], v[62:65], v[50:53]
	v_mfma_f32_16x16x32_f16 v[50:53], a[24:27], v[62:65], v[58:61]
	v_mfma_f32_16x16x32_f16 v[58:61], a[32:35], v[62:65], v[70:73]
	v_mfma_f32_16x16x32_f16 v[70:73], a[12:15], a[40:43], v[82:85]
	v_mfma_f32_16x16x32_f16 v[82:85], a[24:27], a[40:43], v[100:103]
	s_nop 2
	v_accvgpr_read_b32 v102, a154
	v_accvgpr_read_b32 v103, a155
	v_accvgpr_read_b32 v104, a156
	v_accvgpr_read_b32 v105, a157
	v_mfma_f32_16x16x32_f16 v[0:3], a[8:11], v[30:33], v[0:3]
	v_mfma_f32_16x16x32_f16 v[30:33], a[36:39], v[30:33], v[38:41]
	v_mfma_f32_16x16x32_f16 v[38:41], a[12:15], v[62:65], v[46:49]
	v_mfma_f32_16x16x32_f16 v[46:49], a[20:23], v[62:65], v[54:57]
	v_mfma_f32_16x16x32_f16 v[54:57], a[28:31], v[62:65], v[66:69]
	v_mfma_f32_16x16x32_f16 v[66:69], a[8:11], a[40:43], v[78:81]
	v_mfma_f32_16x16x32_f16 v[78:81], a[20:23], a[40:43], v[90:93]
	v_mfma_f32_16x16x32_f16 v[90:93], a[32:35], a[40:43], v[116:119]
	v_mfma_f32_16x16x32_f16 v[116:119], a[16:19], a[44:47], v[136:139]
	v_mfma_f32_16x16x32_f16 v[136:139], a[28:31], a[44:47], v[152:155]
	v_mfma_f32_16x16x32_f16 v[152:155], a[8:11], a[48:51], v[176:179]
	v_mfma_f32_16x16x32_f16 v[176:179], a[20:23], a[48:51], v[196:199]
	v_mfma_f32_16x16x32_f16 v[196:199], a[32:35], a[48:51], v[208:211]
	v_mfma_f32_16x16x32_f16 v[208:211], a[16:19], a[52:55], v[220:223]
	v_mfma_f32_16x16x32_f16 v[220:223], a[28:31], a[52:55], v[102:105]
	s_nop 2
	v_accvgpr_read_b32 v102, a150
	v_accvgpr_read_b32 v103, a151
	v_accvgpr_read_b32 v104, a152
	v_accvgpr_read_b32 v105, a153
	v_mfma_f32_16x16x32_f16 a[134:137], a[12:15], a[56:59], a[134:137]
	v_mfma_f32_16x16x32_f16 a[142:145], a[16:19], a[56:59], a[142:145]
	v_mfma_f32_16x16x32_f16 v[112:115], a[20:23], a[56:59], v[112:115]
	v_mfma_f32_16x16x32_f16 v[120:123], a[24:27], a[56:59], v[120:123]
	v_mfma_f32_16x16x32_f16 v[128:131], a[28:31], a[56:59], v[128:131]
	v_mfma_f32_16x16x32_f16 v[248:251], a[8:11], a[56:59], v[248:251]
	v_mfma_f32_16x16x32_f16 v[232:235], a[8:11], a[52:55], v[232:235]
	v_mfma_f32_16x16x32_f16 v[62:65], a[36:39], v[62:65], v[74:77]
	v_mfma_f32_16x16x32_f16 v[74:77], a[16:19], a[40:43], v[86:89]
	v_mfma_f32_16x16x32_f16 v[86:89], a[28:31], a[40:43], v[108:111]
	v_mfma_f32_16x16x32_f16 v[94:97], a[36:39], a[40:43], v[124:127]
	v_mfma_f32_16x16x32_f16 a[138:141], a[8:11], a[44:47], a[138:141]
	v_mfma_f32_16x16x32_f16 v[108:111], a[12:15], a[44:47], v[132:135]
	v_mfma_f32_16x16x32_f16 v[124:127], a[20:23], a[44:47], v[140:143]
	v_mfma_f32_16x16x32_f16 v[132:135], a[24:27], a[44:47], v[148:151]
	v_mfma_f32_16x16x32_f16 v[140:143], a[32:35], a[44:47], v[160:163]
	v_mfma_f32_16x16x32_f16 v[148:151], a[36:39], a[44:47], v[168:171]
	v_mfma_f32_16x16x32_f16 v[160:163], a[12:15], a[48:51], v[184:187]
	v_mfma_f32_16x16x32_f16 v[168:171], a[16:19], a[48:51], v[192:195]
	v_mfma_f32_16x16x32_f16 v[184:187], a[24:27], a[48:51], v[200:203]
	v_mfma_f32_16x16x32_f16 v[192:195], a[28:31], a[48:51], v[204:207]
	v_mfma_f32_16x16x32_f16 v[200:203], a[36:39], a[48:51], v[212:215]
	v_mfma_f32_16x16x32_f16 v[204:207], a[12:15], a[52:55], v[216:219]
	v_mfma_f32_16x16x32_f16 v[212:215], a[20:23], a[52:55], v[224:227]
	v_mfma_f32_16x16x32_f16 v[216:219], a[24:27], a[52:55], v[228:231]
	v_mfma_f32_16x16x32_f16 v[224:227], a[32:35], a[52:55], v[240:243]
	v_mfma_f32_16x16x32_f16 v[228:231], a[36:39], a[52:55], v[236:239]
	v_mfma_f32_16x16x32_f16 v[236:239], a[32:35], a[56:59], v[102:105]
	v_mfma_f32_16x16x32_f16 v[240:243], a[36:39], a[56:59], v[252:255]
	s_nop 1
	v_accvgpr_read_b32 v102, a146
	v_accvgpr_read_b32 v103, a147
	v_accvgpr_read_b32 v104, a148
	v_accvgpr_read_b32 v105, a149
	s_waitcnt lgkmcnt(7)
	v_mfma_f32_16x16x32_f16 v[0:3], a[60:63], a[88:91], v[0:3]
	s_waitcnt lgkmcnt(1)
	v_mfma_f32_16x16x32_f16 a[134:137], a[64:67], a[112:115], a[134:137]
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_16x16x32_f16 v[156:159], a[68:71], a[4:7], v[156:159]
	v_mfma_f32_16x16x32_f16 v[164:167], a[72:75], a[4:7], v[164:167]
	v_mfma_f32_16x16x32_f16 v[172:175], a[76:79], a[4:7], v[172:175]
	v_mfma_f32_16x16x32_f16 v[180:183], a[80:83], a[4:7], v[180:183]
	v_mfma_f32_16x16x32_f16 v[188:191], a[84:87], a[4:7], v[188:191]
	v_mfma_f32_16x16x32_f16 v[144:147], a[60:63], a[4:7], v[144:147]
	v_mfma_f32_16x16x32_f16 a[142:145], a[68:71], a[112:115], a[142:145]
	v_mfma_f32_16x16x32_f16 v[112:115], a[72:75], a[112:115], v[112:115]
	v_mfma_f32_16x16x32_f16 v[120:123], a[76:79], a[112:115], v[120:123]
	v_mfma_f32_16x16x32_f16 v[128:131], a[80:83], a[112:115], v[128:131]
	v_mfma_f32_16x16x32_f16 v[232:235], a[60:63], a[108:111], v[232:235]
	v_mfma_f32_16x16x32_f16 v[244:247], a[64:67], a[4:7], v[244:247]
	v_mfma_f32_16x16x32_f16 v[6:9], a[64:67], a[88:91], v[6:9]
	v_mfma_f32_16x16x32_f16 v[10:13], a[68:71], a[88:91], v[10:13]
	v_mfma_f32_16x16x32_f16 v[14:17], a[72:75], a[88:91], v[14:17]
	v_mfma_f32_16x16x32_f16 v[18:21], a[76:79], a[88:91], v[18:21]
	v_mfma_f32_16x16x32_f16 v[22:25], a[80:83], a[88:91], v[22:25]
	v_mfma_f32_16x16x32_f16 v[26:29], a[84:87], a[88:91], v[26:29]
	v_mfma_f32_16x16x32_f16 v[30:33], a[0:3], a[88:91], v[30:33]
	v_mfma_f32_16x16x32_f16 v[34:37], a[60:63], a[92:95], v[34:37]
	v_mfma_f32_16x16x32_f16 v[38:41], a[64:67], a[92:95], v[38:41]
	v_mfma_f32_16x16x32_f16 v[42:45], a[68:71], a[92:95], v[42:45]
	v_mfma_f32_16x16x32_f16 v[46:49], a[72:75], a[92:95], v[46:49]
	v_mfma_f32_16x16x32_f16 v[50:53], a[76:79], a[92:95], v[50:53]
	v_mfma_f32_16x16x32_f16 v[54:57], a[80:83], a[92:95], v[54:57]
	v_mfma_f32_16x16x32_f16 v[58:61], a[84:87], a[92:95], v[58:61]
	v_mfma_f32_16x16x32_f16 v[62:65], a[0:3], a[92:95], v[62:65]
	v_mfma_f32_16x16x32_f16 v[66:69], a[60:63], a[96:99], v[66:69]
	v_mfma_f32_16x16x32_f16 v[70:73], a[64:67], a[96:99], v[70:73]
	v_mfma_f32_16x16x32_f16 v[74:77], a[68:71], a[96:99], v[74:77]
	v_mfma_f32_16x16x32_f16 v[78:81], a[72:75], a[96:99], v[78:81]
	v_mfma_f32_16x16x32_f16 v[82:85], a[76:79], a[96:99], v[82:85]
	v_mfma_f32_16x16x32_f16 v[86:89], a[80:83], a[96:99], v[86:89]
	v_mfma_f32_16x16x32_f16 v[90:93], a[84:87], a[96:99], v[90:93]
	v_mfma_f32_16x16x32_f16 v[94:97], a[0:3], a[96:99], v[94:97]
	v_mfma_f32_16x16x32_f16 a[138:141], a[60:63], a[100:103], a[138:141]
	v_mfma_f32_16x16x32_f16 v[108:111], a[64:67], a[100:103], v[108:111]
	v_mfma_f32_16x16x32_f16 v[116:119], a[68:71], a[100:103], v[116:119]
	v_mfma_f32_16x16x32_f16 v[124:127], a[72:75], a[100:103], v[124:127]
	v_mfma_f32_16x16x32_f16 v[132:135], a[76:79], a[100:103], v[132:135]
	v_mfma_f32_16x16x32_f16 v[136:139], a[80:83], a[100:103], v[136:139]
	v_mfma_f32_16x16x32_f16 v[140:143], a[84:87], a[100:103], v[140:143]
	v_mfma_f32_16x16x32_f16 v[148:151], a[0:3], a[100:103], v[148:151]
	v_mfma_f32_16x16x32_f16 v[152:155], a[60:63], a[104:107], v[152:155]
	v_mfma_f32_16x16x32_f16 v[160:163], a[64:67], a[104:107], v[160:163]
	v_mfma_f32_16x16x32_f16 v[168:171], a[68:71], a[104:107], v[168:171]
	v_mfma_f32_16x16x32_f16 v[176:179], a[72:75], a[104:107], v[176:179]
	v_mfma_f32_16x16x32_f16 v[184:187], a[76:79], a[104:107], v[184:187]
	v_mfma_f32_16x16x32_f16 v[192:195], a[80:83], a[104:107], v[192:195]
	v_mfma_f32_16x16x32_f16 v[196:199], a[84:87], a[104:107], v[196:199]
	v_mfma_f32_16x16x32_f16 v[200:203], a[0:3], a[104:107], v[200:203]
	v_mfma_f32_16x16x32_f16 v[204:207], a[64:67], a[108:111], v[204:207]
	v_mfma_f32_16x16x32_f16 v[208:211], a[68:71], a[108:111], v[208:211]
	v_mfma_f32_16x16x32_f16 v[212:215], a[72:75], a[108:111], v[212:215]
	v_mfma_f32_16x16x32_f16 v[216:219], a[76:79], a[108:111], v[216:219]
	v_mfma_f32_16x16x32_f16 v[220:223], a[80:83], a[108:111], v[220:223]
	v_mfma_f32_16x16x32_f16 v[224:227], a[84:87], a[108:111], v[224:227]
	v_mfma_f32_16x16x32_f16 v[228:231], a[0:3], a[108:111], v[228:231]
	v_mfma_f32_16x16x32_f16 v[248:251], a[60:63], a[112:115], v[248:251]
	v_mfma_f32_16x16x32_f16 v[98:101], a[84:87], a[112:115], v[236:239]
	v_mfma_f32_16x16x32_f16 v[252:255], a[0:3], a[112:115], v[240:243]
	v_mfma_f32_16x16x32_f16 v[104:107], a[0:3], a[4:7], v[102:105]
	s_cbranch_scc1 .LBB0_1
; %bb.2:                                ; %._crit_edge
	v_cvt_pk_f16_f32 v5, v12, v13
	v_cvt_pk_f16_f32 v12, v26, v27
	v_cvt_pk_f16_f32 v27, v32, v33
	v_cvt_pk_f16_f32 v33, v76, v77
	v_cvt_pk_f16_f32 v77, v110, v111
	v_cvt_pk_f16_f32 v110, v112, v113
	v_accvgpr_read_b32 v112, a116
	v_and_b32_e32 v112, 15, v112
	s_lshl_b32 s0, s17, 3
	v_and_or_b32 v112, s0, 16, v112
	s_mul_i32 s0, s18, s3
	s_ashr_i32 s1, s0, 31
	s_lshl_b64 s[0:1], s[0:1], 1
	v_cvt_pk_f16_f32 v4, v10, v11
	v_cvt_pk_f16_f32 v10, v22, v23
	v_cvt_pk_f16_f32 v22, v38, v39
	v_cvt_pk_f16_f32 v38, v62, v63
	v_cvt_pk_f16_f32 v63, v154, v155
	v_cvt_pk_f16_f32 v154, v128, v129
	v_accvgpr_read_b32 v128, a117
	s_add_u32 s2, s12, s0
	v_lshrrev_b32_e32 v128, 2, v128
	s_addc_u32 s4, s13, s1
	s_ashr_i32 s17, s16, 31
	v_and_b32_e32 v128, 28, v128
	s_lshl_b64 s[0:1], s[16:17], 1
	v_cvt_pk_f16_f32 v11, v24, v25
	v_cvt_pk_f16_f32 v25, v36, v37
	v_cvt_pk_f16_f32 v36, v66, v67
	v_cvt_pk_f16_f32 v76, v108, v109
	v_cvt_pk_f16_f32 v66, v140, v141
	v_cvt_pk_f16_f32 v67, v142, v143
	v_cvt_pk_f16_f32 v62, v152, v153
	v_cvt_pk_f16_f32 v111, v114, v115
	v_cvt_pk_f16_f32 v108, v120, v121
	v_cvt_pk_f16_f32 v109, v122, v123
	v_cvt_pk_f16_f32 v155, v130, v131
	v_cvt_pk_f16_f32 v153, v100, v101
	v_cvt_pk_f16_f32 v100, v144, v145
	v_or_b32_e32 v113, 32, v112
	v_or_b32_e32 v114, 64, v112
	v_or_b32_e32 v115, 0x60, v112
	v_or_b32_e32 v120, 0x80, v112
	v_or_b32_e32 v121, 0xa0, v112
	v_or_b32_e32 v122, 0xc0, v112
	v_or_b32_e32 v123, 0xe0, v112
	v_or_b32_e32 v129, 32, v128
	v_or_b32_e32 v130, 64, v128
	v_or_b32_e32 v131, 0x60, v128
	v_or_b32_e32 v140, 0x80, v128
	v_or_b32_e32 v141, 0xa0, v128
	v_or_b32_e32 v142, 0xc0, v128
	v_or_b32_e32 v143, 0xe0, v128
	s_add_u32 s36, s2, s0
	v_mul_lo_u32 v144, v112, s3
	v_cmp_gt_i32_e64 s[28:29], s14, v112
	v_cmp_gt_i32_e64 s[30:31], s15, v128
	v_cvt_pk_f16_f32 v39, v64, v65
	v_cvt_pk_f16_f32 v64, v148, v149
	v_cvt_pk_f16_f32 v65, v150, v151
	v_cvt_pk_f16_f32 v101, v146, v147
	s_addc_u32 s33, s4, s1
	v_mul_lo_u32 v145, v113, s3
	v_mul_lo_u32 v146, v114, s3
	v_mul_lo_u32 v147, v115, s3
	v_mul_lo_u32 v148, v120, s3
	v_mul_lo_u32 v149, v121, s3
	v_mul_lo_u32 v150, v122, s3
	v_mul_lo_u32 v151, v123, s3
	v_cmp_gt_i32_e64 s[26:27], s14, v113
	v_cmp_gt_i32_e64 s[24:25], s14, v114
	v_cmp_gt_i32_e64 s[22:23], s14, v115
	v_cmp_gt_i32_e64 s[20:21], s14, v120
	v_cmp_gt_i32_e64 s[18:19], s14, v121
	v_cmp_gt_i32_e64 s[16:17], s14, v122
	v_cmp_gt_i32_e32 vcc, s14, v123
	v_cmp_gt_i32_e64 s[12:13], s15, v129
	v_cmp_gt_i32_e64 s[10:11], s15, v130
	v_cmp_gt_i32_e64 s[8:9], s15, v131
	v_cmp_gt_i32_e64 s[6:7], s15, v140
	v_cmp_gt_i32_e64 s[4:5], s15, v141
	v_cmp_gt_i32_e64 s[2:3], s15, v142
	v_cmp_gt_i32_e64 s[0:1], s15, v143
	v_add_lshl_u32 v112, v128, v144, 1
	v_bfrev_b32_e32 v113, 1
	s_and_b64 s[14:15], s[28:29], s[30:31]
	v_cvt_pk_f16_f32 v0, v0, v1
	v_cvt_pk_f16_f32 v1, v2, v3
	s_and_b32 s37, s33, 0xffff
	s_mov_b32 s39, 0x27000
	s_mov_b32 s38, 0x7ffffffe
	v_cndmask_b32_e64 v112, v113, v112, s[14:15]
	buffer_store_dwordx2 v[0:1], v112, s[36:39], 0 offen
	v_add_lshl_u32 v0, v129, v144, 1
	s_and_b64 s[14:15], s[28:29], s[12:13]
	v_cvt_pk_f16_f32 v2, v6, v7
	v_cvt_pk_f16_f32 v3, v8, v9
	v_cndmask_b32_e64 v0, v113, v0, s[14:15]
	buffer_store_dwordx2 v[2:3], v0, s[36:39], 0 offen
	v_add_lshl_u32 v0, v130, v144, 1
	s_and_b64 s[14:15], s[28:29], s[10:11]
	v_cndmask_b32_e64 v0, v113, v0, s[14:15]
	buffer_store_dwordx2 v[4:5], v0, s[36:39], 0 offen
	v_add_lshl_u32 v0, v131, v144, 1
	s_and_b64 s[14:15], s[28:29], s[8:9]
	v_cvt_pk_f16_f32 v6, v14, v15
	v_cvt_pk_f16_f32 v7, v16, v17
	v_cndmask_b32_e64 v0, v113, v0, s[14:15]
	buffer_store_dwordx2 v[6:7], v0, s[36:39], 0 offen
	v_add_lshl_u32 v0, v140, v144, 1
	s_and_b64 s[14:15], s[28:29], s[6:7]
	v_cvt_pk_f16_f32 v8, v18, v19
	v_cvt_pk_f16_f32 v9, v20, v21
	v_cndmask_b32_e64 v0, v113, v0, s[14:15]
	buffer_store_dwordx2 v[8:9], v0, s[36:39], 0 offen
	v_add_lshl_u32 v0, v141, v144, 1
	s_and_b64 s[14:15], s[28:29], s[4:5]
	v_cndmask_b32_e64 v0, v113, v0, s[14:15]
	buffer_store_dwordx2 v[10:11], v0, s[36:39], 0 offen
	v_add_lshl_u32 v0, v142, v144, 1
	s_and_b64 s[14:15], s[28:29], s[2:3]
	v_cvt_pk_f16_f32 v13, v28, v29
	v_cndmask_b32_e64 v0, v113, v0, s[14:15]
	buffer_store_dwordx2 v[12:13], v0, s[36:39], 0 offen
	v_add_lshl_u32 v0, v143, v144, 1
	s_and_b64 s[14:15], s[28:29], s[0:1]
	v_cvt_pk_f16_f32 v26, v30, v31
	v_cndmask_b32_e64 v0, v113, v0, s[14:15]
	buffer_store_dwordx2 v[26:27], v0, s[36:39], 0 offen
	v_add_lshl_u32 v0, v145, v128, 1
	s_and_b64 s[14:15], s[26:27], s[30:31]
	v_cvt_pk_f16_f32 v24, v34, v35
	v_cndmask_b32_e64 v0, v113, v0, s[14:15]
	buffer_store_dwordx2 v[24:25], v0, s[36:39], 0 offen
	v_add_lshl_u32 v0, v129, v145, 1
	s_and_b64 s[14:15], s[26:27], s[12:13]
	v_cvt_pk_f16_f32 v23, v40, v41
	v_cndmask_b32_e64 v0, v113, v0, s[14:15]
	buffer_store_dwordx2 v[22:23], v0, s[36:39], 0 offen
	v_add_lshl_u32 v0, v130, v145, 1
	s_and_b64 s[14:15], s[26:27], s[10:11]
	v_cvt_pk_f16_f32 v20, v42, v43
	v_cvt_pk_f16_f32 v21, v44, v45
	v_cndmask_b32_e64 v0, v113, v0, s[14:15]
	buffer_store_dwordx2 v[20:21], v0, s[36:39], 0 offen
	v_add_lshl_u32 v0, v131, v145, 1
	s_and_b64 s[14:15], s[26:27], s[8:9]
	v_cvt_pk_f16_f32 v18, v46, v47
	v_cvt_pk_f16_f32 v19, v48, v49
	v_cndmask_b32_e64 v0, v113, v0, s[14:15]
	buffer_store_dwordx2 v[18:19], v0, s[36:39], 0 offen
	v_add_lshl_u32 v0, v140, v145, 1
	s_and_b64 s[14:15], s[26:27], s[6:7]
	v_cvt_pk_f16_f32 v16, v50, v51
	v_cvt_pk_f16_f32 v17, v52, v53
	v_cndmask_b32_e64 v0, v113, v0, s[14:15]
	buffer_store_dwordx2 v[16:17], v0, s[36:39], 0 offen
	v_add_lshl_u32 v0, v141, v145, 1
	s_and_b64 s[14:15], s[26:27], s[4:5]
	v_cvt_pk_f16_f32 v14, v54, v55
	v_cvt_pk_f16_f32 v15, v56, v57
	v_cndmask_b32_e64 v0, v113, v0, s[14:15]
	buffer_store_dwordx2 v[14:15], v0, s[36:39], 0 offen
	v_add_lshl_u32 v0, v142, v145, 1
	s_and_b64 s[14:15], s[26:27], s[2:3]
	v_cvt_pk_f16_f32 v40, v58, v59
	v_cvt_pk_f16_f32 v41, v60, v61
	v_cndmask_b32_e64 v0, v113, v0, s[14:15]
	buffer_store_dwordx2 v[40:41], v0, s[36:39], 0 offen
	v_add_lshl_u32 v0, v143, v145, 1
	s_and_b64 s[14:15], s[26:27], s[0:1]
	v_cndmask_b32_e64 v0, v113, v0, s[14:15]
	buffer_store_dwordx2 v[38:39], v0, s[36:39], 0 offen
	v_add_lshl_u32 v0, v146, v128, 1
	s_and_b64 s[14:15], s[24:25], s[30:31]
	v_cvt_pk_f16_f32 v37, v68, v69
	v_cndmask_b32_e64 v0, v113, v0, s[14:15]
	buffer_store_dwordx2 v[36:37], v0, s[36:39], 0 offen
	v_add_lshl_u32 v0, v129, v146, 1
	s_and_b64 s[14:15], s[24:25], s[12:13]
	v_cvt_pk_f16_f32 v34, v70, v71
	v_cvt_pk_f16_f32 v35, v72, v73
	v_cndmask_b32_e64 v0, v113, v0, s[14:15]
	buffer_store_dwordx2 v[34:35], v0, s[36:39], 0 offen
	v_add_lshl_u32 v0, v130, v146, 1
	s_and_b64 s[14:15], s[24:25], s[10:11]
	v_cvt_pk_f16_f32 v32, v74, v75
	v_cndmask_b32_e64 v0, v113, v0, s[14:15]
	buffer_store_dwordx2 v[32:33], v0, s[36:39], 0 offen
	v_add_lshl_u32 v0, v131, v146, 1
	s_and_b64 s[14:15], s[24:25], s[8:9]
	v_cvt_pk_f16_f32 v30, v78, v79
	v_cvt_pk_f16_f32 v31, v80, v81
	v_cndmask_b32_e64 v0, v113, v0, s[14:15]
	buffer_store_dwordx2 v[30:31], v0, s[36:39], 0 offen
	v_add_lshl_u32 v0, v140, v146, 1
	s_and_b64 s[14:15], s[24:25], s[6:7]
	v_cvt_pk_f16_f32 v28, v82, v83
	v_cvt_pk_f16_f32 v29, v84, v85
	v_cndmask_b32_e64 v0, v113, v0, s[14:15]
	buffer_store_dwordx2 v[28:29], v0, s[36:39], 0 offen
	v_add_lshl_u32 v0, v141, v146, 1
	s_and_b64 s[14:15], s[24:25], s[4:5]
	v_cvt_pk_f16_f32 v84, v86, v87
	v_cvt_pk_f16_f32 v85, v88, v89
	v_cndmask_b32_e64 v0, v113, v0, s[14:15]
	buffer_store_dwordx2 v[84:85], v0, s[36:39], 0 offen
	v_add_lshl_u32 v0, v142, v146, 1
	s_and_b64 s[14:15], s[24:25], s[2:3]
	v_cvt_pk_f16_f32 v82, v90, v91
	v_cvt_pk_f16_f32 v83, v92, v93
	v_cndmask_b32_e64 v0, v113, v0, s[14:15]
	buffer_store_dwordx2 v[82:83], v0, s[36:39], 0 offen
	v_add_lshl_u32 v0, v143, v146, 1
	s_and_b64 s[14:15], s[24:25], s[0:1]
	v_cvt_pk_f16_f32 v80, v94, v95
	v_cvt_pk_f16_f32 v81, v96, v97
	v_accvgpr_read_b32 v42, a138
	v_cndmask_b32_e64 v0, v113, v0, s[14:15]
	v_accvgpr_read_b32 v43, a139
	v_accvgpr_read_b32 v44, a140
	v_accvgpr_read_b32 v45, a141
	buffer_store_dwordx2 v[80:81], v0, s[36:39], 0 offen
	v_add_lshl_u32 v0, v147, v128, 1
	s_and_b64 s[14:15], s[22:23], s[30:31]
	v_cvt_pk_f16_f32 v78, v42, v43
	v_cvt_pk_f16_f32 v79, v44, v45
	v_cndmask_b32_e64 v0, v113, v0, s[14:15]
	buffer_store_dwordx2 v[78:79], v0, s[36:39], 0 offen
	v_add_lshl_u32 v0, v129, v147, 1
	s_and_b64 s[14:15], s[22:23], s[12:13]
	v_cndmask_b32_e64 v0, v113, v0, s[14:15]
	buffer_store_dwordx2 v[76:77], v0, s[36:39], 0 offen
	v_add_lshl_u32 v0, v130, v147, 1
	s_and_b64 s[14:15], s[22:23], s[10:11]
	v_cvt_pk_f16_f32 v74, v116, v117
	v_cvt_pk_f16_f32 v75, v118, v119
	v_cndmask_b32_e64 v0, v113, v0, s[14:15]
	buffer_store_dwordx2 v[74:75], v0, s[36:39], 0 offen
	v_add_lshl_u32 v0, v131, v147, 1
	s_and_b64 s[14:15], s[22:23], s[8:9]
	v_cvt_pk_f16_f32 v72, v124, v125
	v_cvt_pk_f16_f32 v73, v126, v127
	v_cndmask_b32_e64 v0, v113, v0, s[14:15]
	buffer_store_dwordx2 v[72:73], v0, s[36:39], 0 offen
	v_add_lshl_u32 v0, v140, v147, 1
	s_and_b64 s[14:15], s[22:23], s[6:7]
	v_cvt_pk_f16_f32 v70, v132, v133
	v_cvt_pk_f16_f32 v71, v134, v135
	v_cndmask_b32_e64 v0, v113, v0, s[14:15]
	buffer_store_dwordx2 v[70:71], v0, s[36:39], 0 offen
	v_add_lshl_u32 v0, v141, v147, 1
	s_and_b64 s[14:15], s[22:23], s[4:5]
	v_cvt_pk_f16_f32 v68, v136, v137
	v_cvt_pk_f16_f32 v69, v138, v139
	v_cndmask_b32_e64 v0, v113, v0, s[14:15]
	buffer_store_dwordx2 v[68:69], v0, s[36:39], 0 offen
	v_add_lshl_u32 v0, v142, v147, 1
	s_and_b64 s[14:15], s[22:23], s[2:3]
	v_cndmask_b32_e64 v0, v113, v0, s[14:15]
	buffer_store_dwordx2 v[66:67], v0, s[36:39], 0 offen
	v_add_lshl_u32 v0, v143, v147, 1
	s_and_b64 s[14:15], s[22:23], s[0:1]
	v_cndmask_b32_e64 v0, v113, v0, s[14:15]
	buffer_store_dwordx2 v[64:65], v0, s[36:39], 0 offen
	v_add_lshl_u32 v0, v148, v128, 1
	s_and_b64 s[14:15], s[20:21], s[30:31]
	v_cndmask_b32_e64 v0, v113, v0, s[14:15]
	buffer_store_dwordx2 v[62:63], v0, s[36:39], 0 offen
	v_add_lshl_u32 v0, v129, v148, 1
	s_and_b64 s[14:15], s[20:21], s[12:13]
	v_cvt_pk_f16_f32 v60, v160, v161
	v_cvt_pk_f16_f32 v61, v162, v163
	v_cndmask_b32_e64 v0, v113, v0, s[14:15]
	buffer_store_dwordx2 v[60:61], v0, s[36:39], 0 offen
	v_add_lshl_u32 v0, v130, v148, 1
	s_and_b64 s[14:15], s[20:21], s[10:11]
	v_cvt_pk_f16_f32 v58, v168, v169
	v_cvt_pk_f16_f32 v59, v170, v171
	v_cndmask_b32_e64 v0, v113, v0, s[14:15]
	buffer_store_dwordx2 v[58:59], v0, s[36:39], 0 offen
	v_add_lshl_u32 v0, v131, v148, 1
	s_and_b64 s[14:15], s[20:21], s[8:9]
	v_cvt_pk_f16_f32 v56, v176, v177
	v_cvt_pk_f16_f32 v57, v178, v179
	v_cndmask_b32_e64 v0, v113, v0, s[14:15]
	buffer_store_dwordx2 v[56:57], v0, s[36:39], 0 offen
	v_add_lshl_u32 v0, v140, v148, 1
	s_and_b64 s[14:15], s[20:21], s[6:7]
	v_cvt_pk_f16_f32 v54, v184, v185
	v_cvt_pk_f16_f32 v55, v186, v187
	v_cndmask_b32_e64 v0, v113, v0, s[14:15]
	buffer_store_dwordx2 v[54:55], v0, s[36:39], 0 offen
	v_add_lshl_u32 v0, v141, v148, 1
	s_and_b64 s[14:15], s[20:21], s[4:5]
	v_cvt_pk_f16_f32 v52, v192, v193
	v_cvt_pk_f16_f32 v53, v194, v195
	v_cndmask_b32_e64 v0, v113, v0, s[14:15]
	buffer_store_dwordx2 v[52:53], v0, s[36:39], 0 offen
	v_add_lshl_u32 v0, v142, v148, 1
	s_and_b64 s[14:15], s[20:21], s[2:3]
	v_cvt_pk_f16_f32 v50, v196, v197
	v_cvt_pk_f16_f32 v51, v198, v199
	v_cndmask_b32_e64 v0, v113, v0, s[14:15]
	buffer_store_dwordx2 v[50:51], v0, s[36:39], 0 offen
	v_add_lshl_u32 v0, v143, v148, 1
	s_and_b64 s[14:15], s[20:21], s[0:1]
	v_cvt_pk_f16_f32 v48, v200, v201
	v_cvt_pk_f16_f32 v49, v202, v203
	v_cndmask_b32_e64 v0, v113, v0, s[14:15]
	buffer_store_dwordx2 v[48:49], v0, s[36:39], 0 offen
	v_add_lshl_u32 v0, v149, v128, 1
	s_and_b64 s[14:15], s[18:19], s[30:31]
	v_cvt_pk_f16_f32 v46, v232, v233
	v_cvt_pk_f16_f32 v47, v234, v235
	v_cndmask_b32_e64 v0, v113, v0, s[14:15]
	buffer_store_dwordx2 v[46:47], v0, s[36:39], 0 offen
	v_add_lshl_u32 v0, v129, v149, 1
	s_and_b64 s[14:15], s[18:19], s[12:13]
	v_cvt_pk_f16_f32 v44, v204, v205
	v_cvt_pk_f16_f32 v45, v206, v207
	v_cndmask_b32_e64 v0, v113, v0, s[14:15]
	buffer_store_dwordx2 v[44:45], v0, s[36:39], 0 offen
	v_add_lshl_u32 v0, v130, v149, 1
	s_and_b64 s[14:15], s[18:19], s[10:11]
	v_cvt_pk_f16_f32 v42, v208, v209
	v_cvt_pk_f16_f32 v43, v210, v211
	v_cndmask_b32_e64 v0, v113, v0, s[14:15]
	buffer_store_dwordx2 v[42:43], v0, s[36:39], 0 offen
	v_add_lshl_u32 v0, v131, v149, 1
	s_and_b64 s[14:15], s[18:19], s[8:9]
	v_cvt_pk_f16_f32 v138, v212, v213
	v_cvt_pk_f16_f32 v139, v214, v215
	v_cndmask_b32_e64 v0, v113, v0, s[14:15]
	buffer_store_dwordx2 v[138:139], v0, s[36:39], 0 offen
	v_add_lshl_u32 v0, v140, v149, 1
	s_and_b64 s[14:15], s[18:19], s[6:7]
	v_cvt_pk_f16_f32 v136, v216, v217
	v_cvt_pk_f16_f32 v137, v218, v219
	v_cndmask_b32_e64 v0, v113, v0, s[14:15]
	buffer_store_dwordx2 v[136:137], v0, s[36:39], 0 offen
	v_add_lshl_u32 v0, v141, v149, 1
	s_and_b64 s[14:15], s[18:19], s[4:5]
	v_cvt_pk_f16_f32 v134, v220, v221
	v_cvt_pk_f16_f32 v135, v222, v223
	v_cndmask_b32_e64 v0, v113, v0, s[14:15]
	buffer_store_dwordx2 v[134:135], v0, s[36:39], 0 offen
	v_add_lshl_u32 v0, v142, v149, 1
	s_and_b64 s[14:15], s[18:19], s[2:3]
	v_cvt_pk_f16_f32 v132, v224, v225
	v_cvt_pk_f16_f32 v133, v226, v227
	v_cndmask_b32_e64 v0, v113, v0, s[14:15]
	buffer_store_dwordx2 v[132:133], v0, s[36:39], 0 offen
	v_add_lshl_u32 v0, v143, v149, 1
	s_and_b64 s[14:15], s[18:19], s[0:1]
	v_cvt_pk_f16_f32 v126, v228, v229
	v_cvt_pk_f16_f32 v127, v230, v231
	v_cndmask_b32_e64 v0, v113, v0, s[14:15]
	buffer_store_dwordx2 v[126:127], v0, s[36:39], 0 offen
	v_add_lshl_u32 v0, v150, v128, 1
	s_and_b64 s[14:15], s[16:17], s[30:31]
	v_cvt_pk_f16_f32 v124, v248, v249
	v_cvt_pk_f16_f32 v125, v250, v251
	v_accvgpr_read_b32 v86, a134
	v_cndmask_b32_e64 v0, v113, v0, s[14:15]
	v_accvgpr_read_b32 v87, a135
	v_accvgpr_read_b32 v88, a136
	v_accvgpr_read_b32 v89, a137
	buffer_store_dwordx2 v[124:125], v0, s[36:39], 0 offen
	v_add_lshl_u32 v0, v129, v150, 1
	s_and_b64 s[14:15], s[16:17], s[12:13]
	v_cvt_pk_f16_f32 v118, v86, v87
	v_cvt_pk_f16_f32 v119, v88, v89
	v_accvgpr_read_b32 v86, a142
	v_cndmask_b32_e64 v0, v113, v0, s[14:15]
	v_accvgpr_read_b32 v87, a143
	v_accvgpr_read_b32 v88, a144
	v_accvgpr_read_b32 v89, a145
	buffer_store_dwordx2 v[118:119], v0, s[36:39], 0 offen
	v_add_lshl_u32 v0, v130, v150, 1
	s_and_b64 s[14:15], s[16:17], s[10:11]
	v_cvt_pk_f16_f32 v116, v86, v87
	v_cvt_pk_f16_f32 v117, v88, v89
	v_cndmask_b32_e64 v0, v113, v0, s[14:15]
	buffer_store_dwordx2 v[116:117], v0, s[36:39], 0 offen
	v_add_lshl_u32 v0, v131, v150, 1
	s_and_b64 s[14:15], s[16:17], s[8:9]
	v_cndmask_b32_e64 v0, v113, v0, s[14:15]
	buffer_store_dwordx2 v[110:111], v0, s[36:39], 0 offen
	v_add_lshl_u32 v0, v140, v150, 1
	s_and_b64 s[14:15], s[16:17], s[6:7]
	v_cndmask_b32_e64 v0, v113, v0, s[14:15]
	buffer_store_dwordx2 v[108:109], v0, s[36:39], 0 offen
	v_add_lshl_u32 v0, v141, v150, 1
	s_and_b64 s[14:15], s[16:17], s[4:5]
	v_cndmask_b32_e64 v0, v113, v0, s[14:15]
	buffer_store_dwordx2 v[154:155], v0, s[36:39], 0 offen
	v_add_lshl_u32 v0, v142, v150, 1
	s_and_b64 s[14:15], s[16:17], s[2:3]
	v_cvt_pk_f16_f32 v152, v98, v99
	v_cndmask_b32_e64 v0, v113, v0, s[14:15]
	buffer_store_dwordx2 v[152:153], v0, s[36:39], 0 offen
	v_add_lshl_u32 v0, v143, v150, 1
	s_and_b64 s[14:15], s[16:17], s[0:1]
	v_cvt_pk_f16_f32 v102, v252, v253
	v_cvt_pk_f16_f32 v103, v254, v255
	v_cndmask_b32_e64 v0, v113, v0, s[14:15]
	buffer_store_dwordx2 v[102:103], v0, s[36:39], 0 offen
	v_add_lshl_u32 v0, v151, v128, 1
	s_and_b64 s[14:15], vcc, s[30:31]
	v_cndmask_b32_e64 v0, v113, v0, s[14:15]
	buffer_store_dwordx2 v[100:101], v0, s[36:39], 0 offen
	v_add_lshl_u32 v0, v129, v151, 1
	s_and_b64 s[12:13], vcc, s[12:13]
	v_cvt_pk_f16_f32 v98, v244, v245
	v_cvt_pk_f16_f32 v99, v246, v247
	v_cndmask_b32_e64 v0, v113, v0, s[12:13]
	buffer_store_dwordx2 v[98:99], v0, s[36:39], 0 offen
	v_add_lshl_u32 v0, v130, v151, 1
	s_and_b64 s[10:11], vcc, s[10:11]
	v_cvt_pk_f16_f32 v96, v156, v157
	v_cvt_pk_f16_f32 v97, v158, v159
	v_cndmask_b32_e64 v0, v113, v0, s[10:11]
	buffer_store_dwordx2 v[96:97], v0, s[36:39], 0 offen
	v_add_lshl_u32 v0, v131, v151, 1
	s_and_b64 s[8:9], vcc, s[8:9]
	v_cvt_pk_f16_f32 v94, v164, v165
	v_cvt_pk_f16_f32 v95, v166, v167
	v_cndmask_b32_e64 v0, v113, v0, s[8:9]
	buffer_store_dwordx2 v[94:95], v0, s[36:39], 0 offen
	v_add_lshl_u32 v0, v140, v151, 1
	s_and_b64 s[6:7], vcc, s[6:7]
	v_cvt_pk_f16_f32 v92, v172, v173
	v_cvt_pk_f16_f32 v93, v174, v175
	v_cndmask_b32_e64 v0, v113, v0, s[6:7]
	buffer_store_dwordx2 v[92:93], v0, s[36:39], 0 offen
	v_add_lshl_u32 v0, v141, v151, 1
	s_and_b64 s[4:5], vcc, s[4:5]
	v_cvt_pk_f16_f32 v90, v180, v181
	v_cvt_pk_f16_f32 v91, v182, v183
	v_cndmask_b32_e64 v0, v113, v0, s[4:5]
	buffer_store_dwordx2 v[90:91], v0, s[36:39], 0 offen
	v_add_lshl_u32 v0, v142, v151, 1
	s_and_b64 s[2:3], vcc, s[2:3]
	v_cvt_pk_f16_f32 v88, v188, v189
	v_cvt_pk_f16_f32 v89, v190, v191
	v_cndmask_b32_e64 v0, v113, v0, s[2:3]
	buffer_store_dwordx2 v[88:89], v0, s[36:39], 0 offen
	v_add_lshl_u32 v0, v143, v151, 1
	s_and_b64 vcc, vcc, s[0:1]
	v_cvt_pk_f16_f32 v86, v104, v105
	v_cvt_pk_f16_f32 v87, v106, v107
	v_cndmask_b32_e32 v0, v113, v0, vcc
	buffer_store_dwordx2 v[86:87], v0, s[36:39], 0 offen
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
; codeLenInByte = 7792
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
