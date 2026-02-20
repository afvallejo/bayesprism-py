#!/usr/bin/env Rscript

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 2) {
  stop("Usage: Rscript generate_golden_small.R <upstream_R_dir> <output_dir>")
}

upstream_r_dir <- args[[1]]
output_dir <- args[[2]]
dir.create(output_dir, showWarnings = FALSE, recursive = TRUE)

source(file.path(upstream_r_dir, "classes.R"))
source(file.path(upstream_r_dir, "JointPost_functions.R"))
source(file.path(upstream_r_dir, "new_prism.R"))
source(file.path(upstream_r_dir, "run_gibbs.R"))
source(file.path(upstream_r_dir, "Rcgminu.R"))
source(file.path(upstream_r_dir, "optim_functions_MAP.R"))
source(file.path(upstream_r_dir, "optim_functions_MLE.R"))
source(file.path(upstream_r_dir, "update_reference.R"))
source(file.path(upstream_r_dir, "run_prism.R"))

# Compact deterministic toy data
genes <- c("G1", "G2", "G3", "G4", "G5", "G6")
reference <- matrix(
  c(
    10, 0, 3, 0, 2, 5,
    9, 1, 4, 0, 1, 6,
    1, 8, 0, 4, 5, 1,
    0, 9, 1, 3, 4, 2,
    6, 2, 8, 1, 0, 3,
    5, 1, 7, 2, 1, 4
  ),
  nrow = 6,
  byrow = TRUE
)
rownames(reference) <- paste0("cell-", 1:6)
colnames(reference) <- genes
reference <- reference * 100

cell.type.labels <- c("A", "A", "B", "B", "C", "C")
cell.state.labels <- c("A_s1", "A_s2", "B_s1", "B_s2", "C_s1", "C_s2")

mixture <- matrix(
  c(
    40, 20, 18, 9, 8, 21,
    25, 30, 11, 14, 9, 19
  ),
  nrow = 2,
  byrow = TRUE
)
rownames(mixture) <- c("bulk-1", "bulk-2")
colnames(mixture) <- genes
mixture <- mixture * 100

prism <- new.prism(
  reference = reference,
  input.type = "count.matrix",
  cell.type.labels = cell.type.labels,
  cell.state.labels = cell.state.labels,
  key = NULL,
  mixture = mixture,
  outlier.cut = 1,
  outlier.fraction = 1,
  pseudo.min = 1e-8
)

# Manual deconvolution flow without package namespace assumptions
gibbs.control <- valid.gibbs.control(
  list(chain.length = 1200, burn.in = 600, thinning = 3, n.cores = 1, seed = 17, alpha = 1)
)
opt.control <- valid.opt.control(list(optimizer = "MLE", maxit = 4000, n.cores = 1))

gibbs.idx <- get.gibbs.idx(gibbs.control)
phi.cs <- prism@phi_cellState@phi

gibbs.list.cs <- lapply(1:nrow(prism@mixture), function(n) {
  set.seed(gibbs.control$seed)
  sample.Z.theta_n(
    X_n = prism@mixture[n, ],
    phi = phi.cs,
    alpha = gibbs.control$alpha,
    gibbs.idx = gibbs.idx,
    compute.elbo = FALSE
  )
})

joint.cs <- newJointPost(
  bulkID = rownames(prism@mixture),
  geneID = colnames(prism@mixture),
  cellType = rownames(phi.cs),
  gibbs.list = gibbs.list.cs
)
joint.ct <- mergeK(jointPost.obj = joint.cs, map = prism@map)

psi <- updateReference(
  Z = joint.ct@Z,
  phi_prime = prism@phi_cellType,
  map = prism@map,
  key = prism@key,
  opt.control = opt.control
)

phi.update <- psi@phi

gibbs.list.final <- lapply(1:nrow(prism@mixture), function(n) {
  set.seed(gibbs.control$seed)
  sample.theta_n(
    X_n = prism@mixture[n, ],
    phi = phi.update,
    alpha = gibbs.control$alpha,
    gibbs.idx = gibbs.idx
  )
})

theta.final <- newThetaPost(
  bulkID = rownames(prism@mixture),
  cellType = rownames(phi.update),
  gibbs.list = gibbs.list.final
)

set.seed(gibbs.control$seed)
sample.one <- sample.Z.theta_n(
  X_n = prism@mixture[1, ],
  phi = prism@phi_cellState@phi,
  alpha = gibbs.control$alpha,
  gibbs.idx = gibbs.idx,
  compute.elbo = FALSE
)

gamma.example <- c(0.1, -0.3, 0.2, 0.0, -0.4, 0.3)
transform.example <- transform.phi_t(prism@phi_cellType@phi[1, ], gamma.example)

map.df <- do.call(
  rbind,
  lapply(names(prism@map), function(ct) {
    data.frame(cell_type = ct, cell_state = prism@map[[ct]], stringsAsFactors = FALSE)
  })
)

labels.df <- data.frame(
  row = rownames(reference),
  cell_type_label = cell.type.labels,
  cell_state_label = cell.state.labels,
  stringsAsFactors = FALSE
)

controls.df <- data.frame(
  chain_length = gibbs.control$chain.length,
  burn_in = gibbs.control$burn.in,
  thinning = gibbs.control$thinning,
  seed = gibbs.control$seed,
  alpha = gibbs.control$alpha,
  optimizer = opt.control$optimizer,
  maxit = opt.control$maxit,
  stringsAsFactors = FALSE
)

write.csv(reference, file.path(output_dir, "reference_input.csv"), quote = FALSE)
write.csv(mixture, file.path(output_dir, "mixture_input.csv"), quote = FALSE)
write.csv(prism@phi_cellState@phi, file.path(output_dir, "phi_cell_state.csv"), quote = FALSE)
write.csv(prism@phi_cellType@phi, file.path(output_dir, "phi_cell_type.csv"), quote = FALSE)
write.csv(joint.cs@theta, file.path(output_dir, "theta_first_state.csv"), quote = FALSE)
write.csv(joint.ct@theta, file.path(output_dir, "theta_first_type.csv"), quote = FALSE)
write.csv(theta.final@theta, file.path(output_dir, "theta_final_type.csv"), quote = FALSE)
write.csv(phi.update, file.path(output_dir, "phi_update.csv"), quote = FALSE)
write.csv(matrix(transform.example, nrow = 1), file.path(output_dir, "transform_phi_t.csv"), quote = FALSE)
write.csv(sample.one$Z_n, file.path(output_dir, "sample_one_Z.csv"), quote = FALSE)
write.csv(matrix(sample.one$theta_n, nrow = 1), file.path(output_dir, "sample_one_theta.csv"), quote = FALSE)
write.csv(map.df, file.path(output_dir, "map.csv"), row.names = FALSE, quote = FALSE)
write.csv(labels.df, file.path(output_dir, "labels.csv"), row.names = FALSE, quote = FALSE)
write.csv(controls.df, file.path(output_dir, "controls.csv"), row.names = FALSE, quote = FALSE)

cat("Golden fixtures written to", output_dir, "\n")
