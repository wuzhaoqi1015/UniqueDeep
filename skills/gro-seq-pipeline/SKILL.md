---
name: "pro-seq-expert"
description: "Expert guidance for GRO-Seq/PRO-Seq data analysis pipeline using Snakemake. Invoke when users ask about PRO-Seq analysis, pipeline configuration, running workflows, interpreting results, or troubleshooting."
---

# PRO‑Seq 专家技能

本技能提供基于Snakemake的GRO‑Seq/PRO‑Seq数据分析流程的专家指导。涵盖从质量控制到高级分析的完整工作流，包括暂停分析、增强子鉴定、差异表达、富集分析、WGCNA、STEM、Motif分析、靶基因预测、TF网络构建和新转录本发现等。

## 概述

该流程专为精密运行测序（PRO‑Seq）和全局运行测序（GRO‑Seq）数据设计。它通过14个集成模块处理原始FASTQ文件：

1. **质量控制** – 接头去除、长度过滤、MultiQC报告
2. **比对** – ncRNA过滤、使用STAR进行基因组比对
3. **单碱基信号处理** – 基因体定量、TPM/RPKM计算、相关性/PCA分析、元图谱
4. **暂停分析** – 启动子vs基因体读段计数、暂停指数（PI）计算、转录状态分类
5. **增强子鉴定** – TSS识别、非编码转录本过滤、双向/单向分类、增强子定量
6. **差异分析** – 基于DESeq2的基因和增强子差异表达分析
7. **富集分析** – GO和KEGG通路富集分析
8. **WGCNA分析** – 加权基因共表达网络分析
9. **STEM分析** – 短时间序列表达挖掘器（时序模式分析）
10. **Motif分析** – 基于HOMER的增强子和启动子区域Motif发现
11. **靶基因预测** – 基于增强子-基因空间邻近性和表达相关性的预测
12. **TF网络构建** – TF–增强子–基因三元关系构建与可视化
13. **新转录本发现** – 基于gffcompare的基因间区转录本鉴定
14. **图表转换** – 批量将PDF图表转换为PNG格式

## 快速开始

### 环境设置

1. **Conda环境** – 激活 `gro‑seq`（Python 3.8+、R 4.0+、Snakemake 8.x、STAR、samtools、bedtools、StringTie）
2. **外部软件** – 确保在 `config/config.yaml` 中设置TSScall、CPC2、STEM、HOMER的路径
3. **参考文件** – 准备基因组FASTA、注释GFF3、染色体大小文件（按染色体名称排序的 `chrom.sizes`）以及用于过滤的ncRNA FASTA文件。

### 索引构建

构建两个STAR索引：

```bash
# ncRNA索引（用于过滤）
STAR --runMode genomeGenerate \
    --runThreadN 16 \
    --genomeDir /path/to/star_index/ncRNA \
    --genomeFastaFiles /path/to/ncRNA.fa \
    --genomeSAindexNbases 6

# 基因组索引（带注释）
STAR --runMode genomeGenerate \
    --runThreadN 16 \
    --genomeDir /path/to/star_index/genome \
    --genomeFastaFiles /path/to/genome.fa \
    --sjdbGTFfile /path/to/annotation.gff3 \
    --sjdbGTFtagExonParentTranscript Parent \
    --sjdbOverhang 149 \
    --genomeSAindexNbases 14
```

### 配置

1. 复制配置模板：`cp config/config.template.yaml config/config.yaml`
2. 编辑 `config/config.yaml` – 设置 `workdir`、`result_dir`、`sample_table`、`rawdata_dir`、基因组路径、注释数据库、外部软件路径和比较组。
3. 准备样本表（`group.csv`），包含列：`sample`、`group`、`r1`、`r2`、`Time`、`Treatment`（可选）。

### 运行流程

```bash
# 干运行以检查工作流
snakemake -n run_full

# 测试模式（模块1‑3）
snakemake run_test -c 40

# 完整模式（所有14个模块）
snakemake run_full -c 40

# 运行特定模块（例如差异分析）
snakemake run_diff_analysis -c 40
```

## 配置指南

`config/config.yaml` 中的关键配置项：

- **run_mode**: `"test"` 或 `"full"`
- **workdir**: 中间文件的工作目录
- **result_dir**: 最终输出目录
- **sample_table**: `group.csv` 的路径
- **rawdata_dir**: 包含原始FASTQ文件的目录
- **genome**: FASTA、GFF3和排序后的 `chrom.sizes` 的路径
- **star_index**: ncRNA和基因组STAR索引的路径
- **go_kegg_db**、**go_gmt_file**、**kegg_gmt_file**: 注释数据库
- **stem_jar**、**homer_bin_dir**、**tsscall_script**: 外部软件路径
- **comparisons**: 比较对列表（例如 `"Treatment_vs_Control"`）

## 模块详情

### 1. 质量控制（qc.smk）
- 使用 `fastp` 进行接头去除和质量过滤
- 长度过滤
- MultiQC报告聚合

### 2. 比对（mapping.smk）
- 可选的ncRNA过滤（使用STAR）
- 使用STAR进行基因组比对，输出排序的BAM文件
- BAM索引创建

### 3. 单碱基信号处理（onebase.smk）
- 使用 `bedtools coverage` 进行基因体读段计数
- TPM/RPKM计算
- 样本相关性热图和PCA分析
- TSS和PAS周围的元图谱生成

### 4. 暂停分析（pausing.smk）
- 启动子（TSS ±150 bp）和基因体（TSS+301 bp至TES）定量
- 暂停指数（PI）= 启动子读段数 / 基因体读段数
- 转录状态分类：
  - Type I：活跃非暂停（RPKM ≥ 3，PI < 3 或 q值 > 0.01）
  - Type II：活跃暂停（RPKM ≥ 3，PI ≥ 3，q值 ≤ 0.01）
  - Type III：非活跃暂停（RPKM ≤ 0.5，PI ≥ 3，q值 ≤ 0.01）
  - Type IV：非活跃非暂停（RPKM ≤ 0.5，PI < 3 或 q值 > 0.01）
- 条件间的状态转换分析

### 5. 增强子鉴定（enhancer.smk）
- 使用 `TSScall.py` 进行TSS识别（默认聚类阈值3000 bp）
- 过滤基因±1000 bp内的区域
- StringTie组装非编码转录本
- CPC2编码潜能预测
- 分类为双向（反向链TSS间距≤3000 bp）和单向增强子
- 使用read‑1比对进行增强子定量（RPKM）

### 6. 差异分析（diff_analysis.smk）
- 基于DESeq2的基因和增强子差异表达分析
- 默认阈值：|log2FC| ≥ 0.58（1.5倍变化），padj ≤ 0.05

### 7. 富集分析（enrichment.smk）
- 使用 `clusterProfiler` 进行GO和KEGG富集分析
- 默认阈值：p值 ≤ 0.05，q值 ≤ 0.1

### 8. WGCNA分析（wgcna.smk）
- 加权基因共表达网络构建
- 关键参数：`min_cluster_size=30`、`deep_split=2`、`merge_cut_height=0.25`、自动软阈值选择

### 9. STEM分析（stem.smk）
- 短时间序列表达挖掘器（需要Java和 `stem.jar`）
- 默认：50个模型轮廓，最大单位变化 = 2，显著性水平 = 0.05

### 10. Motif分析（motif.smk）
- 使用HOMER在增强子和启动子区域发现Motif
- 启动子区域：TSS ±500 bp
- 使用多线程的 `findMotifsGenome.pl`

### 11. 靶基因预测（target_gene.smk）
- 基于空间邻近性（默认窗口 = 100 kb）和表达相关性（Spearman ≥ 0.3，FDR ≤ 0.05）预测增强子靶基因
- 将邻近增强子合并为热点区域（间隔 ≤ 12.5 kb）并分配给最近的活跃基因

### 12. TF网络构建（tf_network.smk）
- 使用显著Motif（p < 0.05）和增强子-基因对应关系构建TF–增强子–基因三元关系
- 可视化为同心圆网络图或交互式桑基图

### 13. 新转录本发现（novel_transcript.smk）
- 通过 `gffcompare` 将组装的转录本与参考注释进行比较
- 保留基因间区转录本（class‑code "u"），至少2个外显子，FPKM ≥ 1，长度 ≥ 200 bp

### 14. 图表转换（plot_conversion.smk）
- 批量将PDF图表转换为PNG格式（300 DPI）以便共享

## 常用命令

### 流程执行
```bash
# 干运行
snakemake -n run_full

# 测试运行（模块1‑3）
snakemake run_test -c 40

# 完整运行
snakemake run_full -c 40

# 运行特定模块
snakemake run_diff_analysis -c 40
snakemake run_enrichment -c 40
snakemake run_stem -c 40
snakemake run_target_gene -c 40
snakemake run_tf_network -c 40
snakemake run_novel_transcript -c 40
```

### 监控与重新运行
```bash
# 监控日志
tail -f run.log
tail -f .snakemake/log/*.log

# 强制重新运行所有内容
snakemake run_full -c 40 -F

# 重新运行特定规则
snakemake run_full -c 40 -R differential_analysis

# 从失败处继续运行
snakemake run_full -c 40 --rerun-incomplete
```

### 实用脚本
```bash
# 生成项目信息表（用于报告）
python scripts/generate_project_info.py config/config.yaml output.xlsx

# 将PDF图表转换为PNG
python scripts/convert_plots.py result/ --recursive --dpi 300
```

## 故障排除

### 常见问题

1. **`chrom.sizes` 未排序** – `bedtools complement` 将失败。确保文件按染色体名称排序：
   ```bash
   sort -k1,1 chrom.sizes > chrom.sizes.sorted
   ```

2. **缺少外部软件路径** – 验证 `config/config.yaml` 中的所有路径（stem_jar、homer_bin_dir、tsscall_script）是否正确且可访问。

3. **STAR索引生成** – `--genomeSAindexNbases` 参数应根据基因组大小调整。使用提供的帮助脚本：
   ```bash
   samtools faidx genome.fa
   awk '{sum+=$2} END {n=int(log(sum)/log(2)/2-1); if(n>14) n=14; print "Genome size:", sum, "bp"; print "Recommended genomeSAindexNbases:", n}' genome.fa.fai
   ```

4. **Snakemake内存/线程错误** – 调整 `-c`（核心数）参数并检查资源可用性。

5. **R包安装失败** – 确保所有必需的R包（DESeq2、clusterProfiler、WGCNA、tidyverse等）已安装在 `gro‑seq` Conda环境中。

### 日志文件
- 流程日志：`run.log`（如果使用 `nohup` 或 `>&`）
- Snakemake日志：`.snakemake/log/*.log`
- 规则特定日志：`logs/<模块>/*.log`

## 参考文档

- **本地使用指南**：`docs/LOCAL_USAGE_GUIDE.md`
- **流程命令清单**：`docs/PIPELINE_COMMANDS.md`
- **开发指南**：`docs/DEVELOPMENT_GUIDE.md`
- **Docker指南**：`docker/README.md`
- **问题反馈模板**：`docs/BUG_REPORT_TEMPLATE.md`

---

*技能更新：2026‑02‑26*  
*流程版本：v1.3（2026‑01‑23）*