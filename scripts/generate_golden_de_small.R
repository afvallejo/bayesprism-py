#!/usr/bin/env Rscript

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 2) {
  stop("Usage: Rscript generate_golden_de_small.R <upstream_R_dir> <output_dir>")
}

upstream_r_dir <- args[[1]]
output_dir <- args[[2]]
dir.create(output_dir, showWarnings = FALSE, recursive = TRUE)

suppressPackageStartupMessages({
  library(scran)
  library(BiocParallel)
})

source(file.path(upstream_r_dir, "process_input.R"))

set.seed(2026)
genes <- paste0("G", 1:10)

cell.state.labels <- c(
  rep("A_s1", 4),
  rep("A_s2", 4),
  rep("B_s1", 4),
  rep("B_s2", 3),
  rep("C_s1", 2)
)
cell.type.labels <- ifelse(
  grepl("^A_", cell.state.labels), "A",
  ifelse(grepl("^B_", cell.state.labels), "B", "C")
)
cell.ids <- paste0("cell-", seq_along(cell.state.labels))

state.means <- rbind(
  A_s1 = c(40, 35, 10, 6, 4, 3, 2, 2, 2, 1),
  A_s2 = c(15, 38, 28, 7, 4, 3, 2, 1, 2, 1),
  B_s1 = c(5, 4, 6, 36, 30, 8, 3, 2, 1, 1),
  B_s2 = c(4, 3, 5, 18, 34, 26, 4, 2, 1, 1),
  C_s1 = c(3, 2, 3, 4, 4, 5, 30, 18, 6, 3)
)

sc.dat <- do.call(rbind, lapply(seq_along(cell.state.labels), function(i) {
  lambda <- state.means[cell.state.labels[i], ]
  jitter <- ((i %% 3) - 1) * 0.25
  rpois(length(genes), lambda = lambda + jitter)
}))
sc.dat <- as.matrix(sc.dat)
rownames(sc.dat) <- cell.ids
colnames(sc.dat) <- genes

pseudo.count <- 0.1
cell.count.cutoff <- 3
pval.max <- 0.05
lfc.min <- 0.1

stat <- get.exp.stat(
  sc.dat = sc.dat,
  cell.type.labels = cell.type.labels,
  cell.state.labels = cell.state.labels,
  pseudo.count = pseudo.count,
  cell.count.cutoff = cell.count.cutoff,
  n.cores = 1
)

marker <- select.marker(
  sc.dat = sc.dat,
  stat = stat,
  pval.max = pval.max,
  lfc.min = lfc.min
)

write.csv(sc.dat, file.path(output_dir, "sc_dat.csv"), quote = FALSE)

labels.df <- data.frame(
  cell_id = cell.ids,
  cell_type = cell.type.labels,
  cell_state = cell.state.labels,
  stringsAsFactors = FALSE
)
write.csv(labels.df, file.path(output_dir, "labels.csv"), row.names = FALSE, quote = FALSE)

params.df <- data.frame(
  pseudo_count = pseudo.count,
  cell_count_cutoff = cell.count.cutoff,
  pval_max = pval.max,
  lfc_min = lfc.min,
  stringsAsFactors = FALSE
)
write.csv(params.df, file.path(output_dir, "params.csv"), row.names = FALSE, quote = FALSE)

for (ct in names(stat)) {
  write.csv(stat[[ct]], file.path(output_dir, paste0("stat_", ct, ".csv")), quote = FALSE)
}

write.csv(marker, file.path(output_dir, "marker_matrix.csv"), quote = FALSE)
write.csv(
  data.frame(gene = colnames(marker), stringsAsFactors = FALSE),
  file.path(output_dir, "marker_genes.csv"),
  row.names = FALSE,
  quote = FALSE
)

cat("DE golden fixtures written to", output_dir, "\n")
