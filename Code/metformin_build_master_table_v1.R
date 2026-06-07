############################################################
## Metformin Quality – Master Table Builder (tidyverse R) ##
############################################################
## Inputs  : Redica event logs, Metformin Site-Score table,
##           NDC–manufacturer cross-walk, Valisure tests,
##           IQVIA NPA monthly volume
## Output   : master_table.parquet  (+ CSV preview)
## Requires : tidyverse, readxl, arrow, lubridate, stringr
############################################################

library(tidyverse)
library(readxl)
library(lubridate)
library(arrow)
library(stringr)
library(dplyr)

## 0.---  Project paths -------------------------------------------------------
DATA_ROOT <- "G:/My Drive/North Carolina State University/Project - Drug Shortage/Data/06 - Metformin Data"
OUT_DIR   <- file.path(DATA_ROOT, "derived")
if (!dir.exists(OUT_DIR)) dir.create(OUT_DIR)

## 1.---  Parameters ----------------------------------------------------------
REDICA_START  <- ymd("2018-01-01")
REDICA_END    <- ymd("2021-09-01")
IQVIA_START   <- ymd("2021-01-01")
IQVIA_END     <- ymd("2021-12-31")

## 2.---  Helper utilities ----------------------------------------------------
to_ndc11 <- function(x) {
  # 1. remove everything but digits
  digs <- str_remove_all(as.character(x), "\\D")
  # 2. if too long, keep only the first 11
  digs <- ifelse(str_length(digs) > 11,
                 str_sub(digs, 1, 11),
                 digs)
  # 3. if too short, pad on the left with zeros
  str_pad(digs, width = 11, side = "left", pad = "0")
}

## 3.---  Read & aggregate Redica logs ------------------------------------
library(readxl); library(dplyr); library(lubridate); library(stringr); library(janitor)

# helper to find the event-date column
find_date_col <- function(df) {
  dates <- names(df)[str_detect(names(df), regex("date", ignore_case = TRUE))]
  evt   <- dates[str_detect(dates, regex("event", ignore_case = TRUE))]
  if (length(evt) >= 1) return(evt[1])
  if (length(dates) >= 1) return(dates[1])
  stop("No date-like column in sheet")
}

read_redica_event <- function(fp) {
  site_id <- str_extract(basename(fp), "^[0-9]{6,}") %>% as.integer()
  df <- read_excel(fp) %>% clean_names()
  date_col <- find_date_col(df)
  df %>% 
    rename(event_date = !!sym(date_col)) %>% 
    mutate(site_id    = site_id,
           event_date = as_date(event_date)) %>% 
    select(site_id, event_date, site_score)
}

redica_logs <- list.files(
  DATA_ROOT, pattern = "^[0-9]{6,}.*\\.xlsx?$", full.names = TRUE
) %>% 
  purrr::map_dfr(read_redica_event) %>% 
  filter(between(event_date, REDICA_START, REDICA_END))

# ▸ site-level total for 2018-01-01 → 2021-09-01
redica_total <- redica_logs %>% 
  group_by(site_id) %>% 
  summarise(site_score_redica = sum(site_score, na.rm = TRUE), .groups = "drop")

# ▸ yearly view for trend plots (optional EDA)
redica_yearly <- redica_logs %>% 
  mutate(year = year(event_date)) %>% 
  group_by(site_id, year) %>% 
  summarise(year_score = sum(site_score, na.rm = TRUE), .groups = "drop")

library(ggplot2)

# Bar chart of cumulative site scores
# redica_total = site_id + site_score_redica
ggplot(redica_total, aes(
  x = reorder(factor(site_id), site_score_redica),
  y = site_score_redica
)) +
  geom_col(fill = "steelblue") +
  coord_flip() +
  labs(
    title = "Cumulative Redica Site Scores (2018–2021)",
    x = "Site ID",
    y = "Total Site Score"
  ) +
  theme_minimal()

# Yearly small‐multiples for the top 5 sites
# pick the top 5 by total score
top5_ids <- redica_total %>%
  top_n(5, site_score_redica) %>%
  pull(site_id)

redica_yearly %>%
  filter(site_id %in% top5_ids) %>%
  ggplot(aes(year, year_score)) +
  geom_line(color = "darkred", size = 1) +
  geom_point(color = "darkred", size = 2) +
  facet_wrap(~ site_id, scales = "free_y") +
  labs(
    title = "Yearly Redica Score Trends for Top 5 Sites",
    x = "Year",
    y = "Annual Score"
  ) +
  theme_light()


## 4.---  Master NDC ↔ manufacturer cross-walk -----------------------------

# 4·1  Read the authoritative Site Score table  (1 row = 1 site)
site_table_fp <- list.files(
  DATA_ROOT,
  pattern = "^Metformin Manufacturer Site Score Table.*\\.xlsx$",
  full.names = TRUE
) %>% 
  .[!basename(.) %>% str_starts("~\\$")] %>% 
  first()


site_table <- read_excel(site_table_fp) %>% 
  clean_names() %>% 
  rename(site_id        = site_app_id,
         site_display   = site_display_name,
         fei            = fei,
         redica_score_aggregated = red_flag_score) %>% 
  mutate(fei = as.character(fei))

# 4·2  Read the NDC map and pull in site_id via FEI
ndc_map_fp <- list.files(
  DATA_ROOT, pattern = "NDCs labelers.*Redica.*\\.xlsx",
  full.names = TRUE
)[1]

ndc_map <- read_excel(ndc_map_fp) %>% 
  clean_names() %>% 
  mutate(
    ndc_11            = to_ndc11(ndc),
    fei_number_redica = as.character(fei_number_redica)   # align type
  ) %>% 
  left_join(
    site_table %>% select(site_id, fei, site_display, redica_score_aggregated),
    by = c("fei_number_redica" = "fei")
  )

# 4·3  Merge in the event-log scores (redica_total) if you want both
ndc_map_final <- ndc_map %>% 
  left_join(redica_total, by = "site_id")

ndc_map_final %>% 
  filter(!is.na(site_score_redica)) %>% 
  summarise(
    corr = cor(site_score_redica, redica_score_aggregated, use = "complete.obs")
  )

## 5.  ── Valisure lots  ──────────────────────────────────────────────────
val_fp <- list.files(DATA_ROOT, "Valisure.*Scoring.*\\.xlsx", full.names = TRUE)[1]

val <- read_excel(val_fp) %>% 
  janitor::clean_names() %>% 
  mutate(
    ndc_11    = to_ndc11(ndc),   # our robust helper
    test_date = as_date(exp)
  )

val_ndc <- val %>%                      # summarise one row per NDC
  group_by(ndc_11) %>% 
  summarise(
    ndma_mean   = mean(average_of_ndma_npday, na.rm = TRUE),
    dmf_mean    = mean(average_of_dmf_npday,  na.rm = TRUE),
    val_score   = mean(score, na.rm = TRUE),
    n_lots      = n(),
    .groups = "drop"
  )

## 6.  ── IQVIA monthly scripts ──────────────────────────────────────────
iqvia_fp <- list.files(DATA_ROOT, "Metformin Selected NDCs NPA.*\\.xlsx", full.names = TRUE)[1]
iqvia_raw <- read_excel(iqvia_fp) %>% janitor::clean_names()

iqvia_long <- iqvia_raw %>% 
  pivot_longer(starts_with("t_rx_"), names_to = "col", values_to = "trx") %>% 
  mutate(
    ym       = str_remove(col, "^t_rx_"),                   # "nov_2018"
    period   = parse_date_time(ym, orders = "my"),          # 2018-11-01
    ndc_11   = to_ndc11(ndc)
  ) %>% 
  filter(between(period, IQVIA_START, IQVIA_END))

iqvia_ndc <- iqvia_long %>%            # total TRx 2021 per NDC
  group_by(ndc_11) %>% 
  summarise(trx_volume = sum(trx, na.rm = TRUE), .groups = "drop")

## 7.  ── Join diagnostics  ──────────────────────────────────────────────
# 7·1  how many Valisure NDCs have IQVIA?
match_v_i <- val_ndc %>% left_join(iqvia_ndc, by = "ndc_11") 
cat("Valisure NDCs:", nrow(val_ndc),
    " – matched IQVIA:", sum(!is.na(match_v_i$trx_volume)), "\n")

# 7·2  how many IQVIA NDCs are in our Redica/FEI map?
match_i_m <- iqvia_ndc %>% left_join(ndc_map_final %>% select(ndc_11), by = "ndc_11")
cat("IQVIA NDCs:", nrow(iqvia_ndc),
    " – matched in Redica-map:", sum(!is.na(match_i_m$ndc_11)), "\n")

# 7·3  list a few that failed to match, so you can inspect manually
iqvia_ndc %>% 
  anti_join(ndc_map_final, by = "ndc_11") %>% 
  slice_head(n = 10)

## 8.  ── Build master table (only matched rows)  ─────────────────────────
master_full <- ndc_map_final %>% 
  inner_join(val_ndc,  by = "ndc_11") %>%   # keep only products with lab tests
  inner_join(iqvia_ndc, by = "ndc_11")      # and market volume

cat("Master rows:", nrow(master_full), "\n")

## 9.  ── Quick sanity plots  ─────────────────────────────────────────────
library(ggplot2)

# 9·1 Distribution of TRx
ggplot(master_full, aes(trx_volume)) +
  geom_histogram(bins = 30) +
  scale_x_log10() +
  labs(title = "Total TRx (2021) – matched NDCs", x = "Scripts (log10)", y = "Count")

# 9·2 Valisure score vs Redica risk
ggplot(master_full, aes(site_score_redica, val_score)) +
  geom_point(aes(size = trx_volume), alpha = .7) +
  scale_size(trans = "sqrt") +
  geom_smooth(method = "lm", se = FALSE, colour = "red") +
  labs(x = "Redica Site Score", y = "Valisure Contaminant Score",
       title = "Inspection Risk vs Lab Contaminants (matched NDCs)")

# 9·3 Who dominates volume?
master_full %>% 
  arrange(desc(trx_volume)) %>% 
  slice_head(n = 10) %>% 
  select(ndc_11, labeler, trx_volume, val_score, site_score_redica)


###############################################################################
## EXTRA EDA – overlap checks & richer visualisations                        ##
###############################################################################

## A.  Dataset Overlap audit  -----------------------------------------------
all_keys <- list(
  redica  = ndc_map_final %>% pull(ndc_11) %>% unique(),
  valisure= val_ndc %>% pull(ndc_11) %>% unique(),
  iqvia   = iqvia_ndc %>% pull(ndc_11) %>% unique()
)

# counts
sapply(all_keys, length)

# Venn-like summary
library(fuzzyjoin)  # install once if needed
overlap_tbl <- expand.grid(
  redica = c(TRUE, FALSE),
  val    = c(TRUE, FALSE),
  iqvia  = c(TRUE, FALSE)
) %>% 
  rowwise() %>% 
  mutate(
    n = length(
      intersect(
        if (redica) all_keys$redica else character(),
        intersect(
          if (val)     all_keys$valisure else character(),
          if (iqvia)   all_keys$iqvia   else character()
        )
      )
    )
  )
print(overlap_tbl)

## B.  Distributions & density curves  --------------------------------------
# Valisure composite score (hist + density)
ggplot(master_full, aes(val_score)) +
  geom_histogram(aes(y = after_stat(density)), bins = 30, fill = "grey70") +
  geom_density(color = "steelblue", linewidth = 1) +
  labs(title = "Valisure Composite Score – density", x = "Score", y = "Density")

# TRx volume on log10 scale (hist + density)
ggplot(master_full, aes(log10(trx_volume + 1))) +
  geom_histogram(aes(y = after_stat(density)), bins = 40, fill = "grey80") +
  geom_density(color = "tomato", linewidth = 1) +
  labs(title = "Scripts (log10) – density", x = "log10(TRx + 1)", y = "Density")

## C.  Total TRx by labeler & by site  --------------------------------------
# top 15 labelers
master_full %>% 
  group_by(labeler) %>% 
  summarise(total_trx = sum(trx_volume), .groups = "drop") %>% 
  arrange(desc(total_trx)) %>% 
  slice_head(n = 15) %>% 
  ggplot(aes(reorder(labeler, total_trx), total_trx/1e6)) +
  geom_col(fill = "forestgreen") +
  coord_flip() +
  labs(title = "Top 15 Labelers by 2021 Scripts",
       x = "", y = "Scripts (millions)") +
  theme_minimal()

# top 15 sites
master_full %>% 
  group_by(site_display) %>% 
  summarise(total_trx = sum(trx_volume), .groups = "drop") %>% 
  arrange(desc(total_trx)) %>% 
  slice_head(n = 15) %>% 
  ggplot(aes(reorder(site_display, total_trx), total_trx/1e6)) +
  geom_col(fill = "steelblue") +
  coord_flip() +
  labs(title = "Top 15 Manufacturing Sites by 2021 Scripts",
       x = "", y = "Scripts (millions)") +
  theme_minimal()

## D.  NDMA vs DMF scatter  --------------------------------------------------
ggplot(master_full, aes(ndma_mean, dmf_mean)) +
  geom_point(aes(size = trx_volume, color = val_score), alpha = .7) +
  scale_size(trans = "sqrt") +
  scale_color_viridis_c(option = "G") +
  labs(title = "NDMA vs DMF (size = TRx, colour = Valisure Score)",
       x = "NDMA mean (ng per max daily dose)",
       y = "DMF mean (ng per max daily dose)",
       colour = "Valisure\nScore",
       size   = "TRx volume") +
  theme_minimal()

## E.  Redica inspection heatmap (site × year)  -----------------------------
heat <- redica_yearly %>% 
  left_join(site_table %>% select(site_id, site_display), by = "site_id")

ggplot(heat, aes(factor(year), reorder(site_display, site_id), fill = year_score)) +
  geom_tile() +
  scale_fill_gradient2(low = "navy", mid = "white", high = "red",
                       midpoint = 0) +
  labs(title = "Redica Annual Score Heatmap",
       x = "Year", y = "Site", fill = "Score") +
  theme_minimal()

## F.  Waterfall of positive vs negative flags  -----------------------------
pos_neg <- redica_logs %>% 
  mutate(flag_type = if_else(site_score > 0, "Negative\n(issues)", "Positive\n(clean)")) %>% 
  group_by(site_id, flag_type) %>% 
  summarise(events = n(), .groups = "drop") %>% 
  left_join(site_table %>% select(site_id, site_display), by = "site_id")

ggplot(pos_neg, aes(events, reorder(site_display, site_id), fill = flag_type)) +
  geom_col(position = "stack") +
  labs(title = "Inspection Events 2018-2021",
       x = "Number of Events", y = "Site", fill = "") +
  theme_minimal()

## G.  Quick summary table  --------------------------------------------------
summary_tbl <- master_full %>% 
  summarise(
    matched_ndcs        = n(),
    pct_with_valisure   = mean(!is.na(val_score)) * 100,
    pct_with_iqvia      = mean(!is.na(trx_volume)) * 100,
    median_val_score    = median(val_score, na.rm = TRUE),
    median_trx_volume   = median(trx_volume, na.rm = TRUE)
  )
print(summary_tbl)

# 9.4 Scatter: Valisure vs Redica
ggplot(master_full, aes(redica_score_aggregated, val_score_mean)) +
  geom_point(aes(size = trx_volume), alpha = .6) +
  geom_smooth(method = "lm", se = FALSE, color = "darkred") +
  scale_size_continuous(trans = "sqrt", name = "TRx Vol") +
  labs(title = "Contaminant Score vs Inspection Risk",
       x = "Redica Score (2018–21)", y = "Valisure Mean Score")

## 10.  ── (Optional) write output  ─────────────────────────────────────—
write_parquet(master_full, file.path(OUT_DIR, "master_full.parquet"))


# ## 5.---  Valisure contaminants ----------------------------------------------
# val_fp <- list.files(
#   DATA_ROOT, pattern = "Valisure.*Scoring.*\\.xlsx", full.names = TRUE
# )[1]
# 
# val_raw <- read_excel(val_fp) %>%
#   clean_names() %>%
#   mutate(
#     ndc_11    = to_ndc11(ndc),
#     test_date = as_date(exp)    # if `exp` is actually the test date
#   )
# 
# # Summarise by NDC:
# val_ndc_summary <- val_raw %>%
#   group_by(ndc_11) %>%
#   summarise(
#     ndma_mean      = mean(average_of_ndma_npday, na.rm = TRUE),
#     dmf_mean       = mean(average_of_dmf_npday, na.rm = TRUE),
#     val_score_mean = mean(score, na.rm = TRUE),
#     n_lots         = n(),
#     .groups = "drop"
#   )
# 
# # (Optional) summarise up to site level
# val_site_summary <- val_ndc_summary %>%
#   left_join(ndc_map_final %>% select(ndc_11, site_id), by = "ndc_11") %>%
#   group_by(site_id) %>%
#   summarise(
#     ndma_site  = weighted.mean(ndma_mean, n_lots, na.rm = TRUE),
#     dmf_site   = weighted.mean(dmf_mean, n_lots, na.rm = TRUE),
#     score_site = weighted.mean(val_score_mean, n_lots, na.rm = TRUE),
#     total_lots = sum(n_lots),
#     .groups    = "drop"
#   )
# 
# # If you want one joined table:
# master1 <- ndc_map_final %>%
#   left_join(val_ndc_summary, by = "ndc_11")
# 
# ## 6.---  IQVIA NPA (utilisation) ---------------------------------------------
# iqvia_fp <- list.files(
#   DATA_ROOT, pattern = "Metformin Selected NDCs NPA.*\\.xlsx", full.names = TRUE
# )[1]
# 
# iqvia_raw <- read_excel(iqvia_fp) %>% clean_names()
# 
# # melt T_Rx_* columns (they start "t_rx_") and parse period
# 
# iqvia_long <- iqvia_raw %>%
#   pivot_longer(
#     cols      = starts_with("t_rx_"),
#     names_to  = "col",
#     values_to = "trx"
#   ) %>%
#   mutate(
#     # strip the prefix and split into two pieces: month, year
#     ym = str_remove(col, "^t_rx_"),    # e.g. "nov_2018"
#     month = str_extract(ym, "^[a-z]+"),# "nov"
#     year  = str_extract(ym, "\\d{4}$"),# "2018"
#     # build a proper ISO string "2018-11-01"
#     period = ymd(paste0(year, "-", 
#                         match(tolower(month), 
#                               tolower(month.abb)) %>% str_pad(2, "left", "0"),
#                         "-01"))
#   ) %>%
#   filter(between(period, IQVIA_START, IQVIA_END))
# 
# # now you should have real rows:
# iqvia_long %>% slice(1:10)
# 
# # Summarise to get total volume per NDC
# iqvia_vol <- iqvia_long %>%
#   group_by(ndc) %>%
#   summarise(
#     trx_volume = sum(trx, na.rm = TRUE),
#     .groups    = "drop"
#   ) %>%
#   mutate(ndc_11 = to_ndc11(ndc))
# 
# # If you want monthly detail for trend plots:
# iqvia_monthly <- iqvia_long %>%
#   mutate(ndc_11 = to_ndc11(ndc)) %>%
#   select(ndc_11, period, trx)
# 
# # And one joined table (with Valisure + IQVIA):
# master_full <- master1 %>%
#   left_join(iqvia_vol, by = "ndc_11")
# 
# # 7.--- Quick EDA after the new joins -----------------------------------------
# 
# # 1. Distribution of composite Valisure score
# ggplot(master_full, aes(val_score_mean)) +
#   geom_histogram(bins = 20, fill = "skyblue") +
#   labs(title = "Valisure Composite Score Distribution",
#        x = "Mean Score per NDC", y = "Count")
# 
# # 2. Distribution of IQVIA volume (log scale)
# ggplot(master_full, aes(trx_volume)) +
#   geom_histogram(bins = 50) +
#   # scale_x_log10() +
#   labs(title = "IQVIA TRx Volume (log scale)",
#        x = "Total TRx (log10)", y = "Count")
# 

# 
# # 4. Boxplot of NDMA by formulation
# ggplot(master_full, aes(formulation, ndma_mean)) +
#   geom_boxplot() +
#   coord_flip() +
#   labs(title = "NDMA Levels by Formulation",
#        x = "Formulation", y = "NDMA Mean (ppb)")
# 
# # 5. Monthly TRx trends (all NDCs)
# ggplot(iqvia_monthly, aes(period, trx, group = ndc_11)) +
#   geom_line(alpha = 1) +
#   facet_wrap(~ndc_11, scales = "free_y") +
#   labs(title = "Monthly Dispensed Scripts by NDC",
#        x = "Month", y = "Scripts") +
#   theme_minimal()
# 
# ## 7.---  Save outputs --------------------------------------------------------
# out_parquet <- file.path(OUT_DIR, "master_table.parquet")
# write_parquet(master, out_parquet)
# 
# write_csv(head(master, 500), file.path(OUT_DIR, "master_preview_sample.csv"))
# 
# cat(glue::glue("\n✅  Master table: {out_parquet}  ({nrow(master)} rows)\n",
#                "⚠︎  First 500 rows also saved as CSV for quick peek.\n"))
