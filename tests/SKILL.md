---
name: gro-seq-pipeline
description: GRO/PRO-Seq 分析流程技能：质控→比对→Onebase→增强子→差异→富集→网络等，基于 Snakemake 与 R/Python 脚本，支持 test/full 与单板块运行
---

# gro-seq-pipeline

面向 GRO-Seq/PRO-Seq 的端到端分析技能。基于仓库内的 Snakemake 流程与 R/Python 脚本，覆盖数据质控、比对、基因表达与 Onebase 定量、增强子/基因间区识别与定量、差异表达、富集分析、WGCNA、STEM、TF 网络与报告汇总。

## 使用场景

- 需要按标准流程运行 GRO/PRO-Seq 全流程或其中一个板块
- 需要讲解每个步骤的输入输出、依赖和注意事项
- 需要快速定位关键脚本与配置，并给出可运行命令
- 需要根据差异结果继续做富集/WGCNA/STEM/暂停指数等分析

## 先决条件

- 已复制 config/config.template.yaml 为 config/config.yaml，并填入绝对路径：
  - 基因组与注释：genome.{fasta,gtf/gff3,chrom_size}
  - 样本与原始数据：sample_table（CSV）、rawdata_dir（FASTQ 根目录）
  - 外部程序：TSScall.py、CPC2.py、stem.jar 等（如使用对应模块）
- 已安装流程依赖（建议 conda 或 Docker）：
  - 命令行：STAR、samtools、bedtools、StringTie、gffread、fastp、fastqc、multiqc、bamCoverage、dot
  - R 包：DESeq2、clusterProfiler、WGCNA、tidyverse、ggplot2、pheatmap、PCAtools 等
- comparisons 与 sample_table 中的分组名保持一致；run_mode 设为 test 或 full

## 关键概念与模块依赖

- 总控：Snakefile 引入各规则文件（rules/*.smk），通过 run_test/run_full 或单板块规则运行
- 数据流：QC → Mapping → Onebase → Pausing/Enhancer → Differential → Enrichment/WGCNA/STEM/Network → 报告
- 差异分析输入：
  - 基因层：Onebase 合并矩阵 gene_counts.txt
  - 增强子层：enhancer 模块产出的 intergenic_{bidir,unidir} 计数矩阵
  - 分组与比较：config.sample_table 与 config.comparisons

## 常用运行命令

- 试测（板块1–3）：snakemake run_test -c <cores>
- 全流程（板块1–全部）：snakemake run_full -c <cores>
- 单板块快捷（示例，更多见 Snakefile）：
  - 质控：snakemake run_qc -c <cores>
  - 比对：snakemake run_mapping -c <cores>
  - Onebase：snakemake run_onebase -c <cores>
  - 增强子：snakemake run_enhancer -c <cores>
  - 差异：snakemake run_diff_analysis -c <cores>
  - 富集：snakemake run_enrichment -c <cores>
  - WGCNA：snakemake run_wgcna -c <cores>
  - STEM：snakemake run_stem -c <cores>
- 生成流程图（需要 graphviz/dot）：snakemake generate_dag

## 板块说明（要点）

1) 质控（rules/qc.smk）
- fastp 去接头/质量过滤，FastQC/MultiQC 汇总
- 输出：过滤后 fq.gz、MultiQC 报告、qc_stats.xlsx

2) 比对（rules/mapping.smk）
- STAR 比对，可选 deRNA 预处理；生成 BAM/BAI 与 BigWig
- 输出：BAM/BAI、STAR 日志、BigWig、mapping_stats.xlsx

3) Onebase 定量（rules/onebase.smk）
- 最长转录本提取 → onebase（5’端）信号 → 基因计数合并 → TPM/RPKM → 相关性/PCA/区域分布/Metaplot
- 输出：gene_counts.txt、表达矩阵与图表、Excel 汇总

4) 暂停分析（rules/pausing.smk）
- Promoter/genebody 划分 → Promoter 滑窗峰值 → 计算 PI（RPKM/TPM）→ 类型分类与状态转换可视化
- 输出：分类/转换表与 PDF/XLSX

5) 增强子与基因间区（rules/enhancer.smk）
- 合并 BAM → TSScall 识别 TSS → 过滤基因区 → StringTie 组装与 CPC2 → 单/双向分类与定量 → 表达分布
- 输出：intergenic_* BED/XLSX、count_matrix（总/单/双向）与分布图

6) 差异分析（rules/diff_analysis.smk + scripts/differential_analysis.R）
- 基因与增强子分别用 DESeq2；按 comparisons 输出 All/sig、火山图/热图/柱图与 xlsx；可选添加注释

7) 富集分析（rules/enrichment.smk）
- 对差异显著集与多来源基因集做 GO/KEGG ORA，输出 xlsx 与点图

8) WGCNA（rules/wgcna.smk）与 STEM（rules/stem.smk）
- 基于表达矩阵与差异结果进行模块挖掘与时序模式分析，输出 PDF/CSV/XLSX

## 关键输入与参数（config/config.yaml）

- 路径：workdir、result_dir、genome.{fasta,gtf,chrom_size}、sample_table、rawdata_dir
- 运行：run_mode（test/full）、assay_type（GRO-Seq/PRO-Seq）、expression_matrix（rpkm/tpm）
- 模块：mapping.enable_deRNA/index、pausing 阈值、bidirection_thresholds/default_bidirection_threshold
- 外部：go/kegg gmt 与注释、TSScall/CPC2 路径、stem_jar 等

## 产物与位置

- 结果位于 result_dir 下的各模块目录：qc、mapping、onebase、enhancer、diff、enrichment、wgcna、stem 等
- 汇总表与图形按模块归档，脚本会生成 xlsx/pdf/图像文件便于报告

## 故障排查

- GTF/GFF 唯一性错误：使用 scripts/dedup_gene_gff.py 预处理后更新 config
- chrom.sizes 需与参考基因组一致并排序
- comparisons 与 sample_table 组名不一致会导致差异分析失败
- TSScall/CPC2/STEM 未配置绝对路径会导致增强子或 STEM 模块跳过或错误

## 示例流程

1) 准备配置
- 复制 config/config.template.yaml 为 config/config.yaml，填入绝对路径
- 按 config/group.template.csv 格式准备 sample_table，并与 comparisons 保持一致

2) 快速试跑
- snakemake run_test -c 8

3) 全流程
- snakemake run_full -c 16

4) 仅做差异与富集（假设已完成 Onebase/Enhancer）
- snakemake run_diff_analysis -c 8
- snakemake run_enrichment -c 8
