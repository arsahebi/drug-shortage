############################################################
##   Comprehensive Descriptive Analysis – Metformin Pilot
############################################################

## 0.  Libraries, paths, helpers -------------------------------------------
library(tidyverse)
library(readxl)
library(lubridate)
library(janitor)
library(ggrepel)

DATA_DIR  <- "G:/My Drive/North Carolina State University/Project - Drug Shortage/Data/06 - Metformin Data"
OUT_DIR   <- file.path(DATA_DIR, "derived")
if (!dir.exists(OUT_DIR)) dir.create(OUT_DIR)

# helper: force numeric string → 11-digit NDC
to_ndc11 <- function(x) {
  digs <- stringr::str_remove_all(as.character(x), "\\D")
  digs <- ifelse(str_length(digs) > 11, str_sub(digs, 1, 11), digs)
  str_pad(digs, 11, "left", "0")
}

## 1.  Redica inspection files ---------------------------------------------
read_inspections <- function(fp) {
  fname <- basename(fp) %>% str_squish()
  rx <- "^(\\d+)\\s*-\\s*(.*?)\\s*\\[\\s*([^_]+?)\\s*_\\s*([^\\]]+?)\\s*\\]"
  m  <- str_match(fname, rx)
  if (is.na(m[1,1])) return(NULL)

  read_excel(fp) %>% clean_names() %>%
    rename(event_date = matches("date", ignore.case = TRUE)) %>% 
    mutate(
      facility_id = m[1,2],
      company     = str_to_title(m[1,3]),
      city        = str_to_title(m[1,4]),
      country     = str_to_title(m[1,5]),
      event_date  = as_date(event_date)
    ) %>%
    select(facility_id, company, city, country,
           event_date, red_flag_criticality, red_flag_type,
           red_flag_value, red_flag_agency, site_score)
}

inspections <- list.files(DATA_DIR, "^[0-9]+.*\\.xlsx?$", full.names = TRUE) %>% 
  discard(~ str_starts(basename(.), "~\\$")) %>% 
  map_dfr(read_inspections)

## 2.  Facility-level risk metrics ------------------------------------------
risk_metrics <- inspections %>% 
  group_by(facility_id, company, city, country) %>% 
  summarise(
    total_inspections    = n(),
    critical_flags       = sum(red_flag_criticality=="Critical", na.rm = TRUE),
    avg_site_score       = mean(site_score, na.rm = TRUE),
    score_volatility     = sd(site_score,   na.rm = TRUE),
    .groups = "drop"
  )

## 3.  NDC → site_id cross-walk ---------------------------------------------
site_table <- read_excel(
  list.files(DATA_DIR, "^Metformin Manufacturer Site Score Table.*\\.xlsx$", 
             full.names = TRUE)[1]) %>% 
  clean_names() %>% 
  transmute(site_id = as.character(site_app_id),
            fei     = as.character(fei))

ndc_raw <- read_excel(
  list.files(DATA_DIR, "NDCs labelers.*Redica.*\\.xlsx", full.names = TRUE)[1]) %>% 
  clean_names()

ndc_map_final <- ndc_raw %>% 
  mutate(ndc_11 = to_ndc11(ndc13)) %>% 
  separate_rows(fei_number_redica, mfr_duns_redica,
                site_display_name_redica, sep = ",\\s*") %>% 
  mutate(fei_number_redica = str_trim(fei_number_redica)) %>% 
  left_join(site_table, by = c("fei_number_redica" = "fei")) %>% 
  distinct(ndc_11, site_id)

## 4.  IQVIA volume 2019-21 --------------------------------------------------
iqvia_long <- read_excel(
  list.files(DATA_DIR, "Metformin Selected NDCs NPA.*\\.xlsx", full.names = TRUE)[1]
) %>% clean_names() %>% 
  pivot_longer(starts_with("t_rx_"), names_to = "col", values_to = "trx") %>% 
  mutate(period = parse_date_time(str_remove(col, "^t_rx_"), "my"),
         ndc_11 = to_ndc11(ndc)) %>% 
  filter(between(period, ymd("2019-01-01"), ymd("2021-12-31")))

iqvia_ndc <- iqvia_long %>% 
  group_by(ndc_11) %>% 
  summarise(trx_volume = sum(trx, na.rm = TRUE), .groups = "drop") %>% 
  left_join(ndc_map_final, by = "ndc_11")

trx_by_site <- iqvia_ndc %>% 
  filter(!is.na(site_id)) %>% 
  group_by(site_id) %>% 
  summarise(trx_all = sum(trx_volume), .groups = "drop")

## 5.  Valisure impurity results (NDC-only merge) ---------------------------
val_fp <- file.path(DATA_DIR,
                    "Valisure data with Mfg name location and ID_Scoring_061424.xlsx")

val_with_id <- read_excel(val_fp) %>% clean_names() %>% 
  mutate(ndc_11 = to_ndc11(ndc13)) %>%                        # **adjust column if needed**
  filter(str_detect(ndc_11, "^\\d{11}$")) %>%                 # keep only valid keys
  left_join(ndc_map_final, by = "ndc_11") %>%                 # attach site_id
  filter(!is.na(site_id))                                     # keep rows that matched

val_site <- val_with_id %>% 
  group_by(site_id) %>% 
  summarise(
    lots_tested = n(),
    ndma_max    = max(ndma_ng_max_daily_dose, na.rm = TRUE),
    dmf_max     = max(dmf_ng_max_daily_dose , na.rm = TRUE),
    .groups = "drop"
  )


## 6.  Bring everything together --------------------------------------------
site_full <- risk_metrics %>%                              # inspections
  left_join(trx_by_site,  by = c("facility_id" = "site_id")) %>%  # + IQVIA
  left_join(val_site,     by = c("facility_id" = "site_id"))      # + Valisure

## 7.  Quick scatter example -------------------------------------------------
ggplot(site_full, aes(avg_site_score, ndma_max)) +
  geom_point(aes(size = trx_all, colour = country), alpha = .7) +
  scale_y_log10() + geom_hline(yintercept = 96, linetype = "dashed") +
  labs(title = "Chemical impurity vs Redica inspection score (Metformin)",
       x = "Average Redica Site Score (2018-21)",
       y = "NDMA max (ng / max daily dose)",
       size = "US TRx 2019-21") +
  theme_minimal()

## 8.  Write outputs ---------------------------------------------------------
write_csv(iqvia_ndc,  file.path(OUT_DIR, "iqvia_enriched_by_ndc.csv"))
write_csv(site_full,  file.path(OUT_DIR, "facility_summary_all_sources.csv"))
write_csv(val_with_id, file.path(OUT_DIR, "valisure_with_siteid.csv"))

############################################################################
##  Valisure vs. “raw” Redica facilities (IDs start with 1000) – SAFE
############################################################################
library(dplyr); library(stringr)

## A.  De-duplicate column names -------------------------------------------
val_with_id <- val_with_id %>% 
  # 1)  keep ONLY the “.x” when both “.x” & “.y” versions exist
  select(-ends_with(".y")) %>% 
  
  # 2)  collapse ‘site_id’ variants FIRST
  mutate(site_id = coalesce(!!! rlang::syms(grep("^site_id", names(.), value = TRUE)))) %>% 
  select(-matches("^site_id\\.(x)$")) %>%   # drop residual suffixed copies
  
  # 3)  strip the remaining “.x” suffixes everywhere else
  rename_with(~ str_replace(., "\\.x$", ""), ends_with(".x"))

## B.  Tag which Valisure rows fall in the “raw” Redica set -----------------
inspections <- inspections %>% mutate(facility_id = as.character(facility_id))

redica_raw_ids <- inspections %>% 
  distinct(facility_id) %>% 
  filter(startsWith(facility_id, "1000")) %>% 
  pull(facility_id)

val_coverage <- val_with_id %>% 
  mutate(site_id       = as.character(site_id),
         in_redica_raw = site_id %in% redica_raw_ids)

## C.  Headline coverage ----------------------------------------------------
cat(
  "\nValisure lots mapping to ‘raw’ Redica facilities:",
  sum(val_coverage$in_redica_raw), "/", nrow(val_coverage),
  "\nUnique Valisure facilities covered:",
  n_distinct(val_coverage$site_id[val_coverage$in_redica_raw]), "/",
  n_distinct(val_coverage$site_id), "\n"
)

## D.  Build PRESENT / MISSING tables with facility names -------------------
fac_lookup <- inspections %>% 
  distinct(facility_id, company, city, country)

val_present <- val_coverage %>% 
  filter(in_redica_raw) %>% 
  left_join(fac_lookup, by = c("site_id" = "facility_id"))

val_missing <- val_coverage %>% 
  filter(!in_redica_raw) %>% 
  left_join(fac_lookup, by = c("site_id" = "facility_id"))

# find whichever columns exist for ‘lot’ and ‘mfr_name’
lot_col      <- names(val_missing)[str_detect(names(val_missing), "^lot")][1]
mfr_name_col <- names(val_missing)[str_detect(names(val_missing), "^mfr_name")][1]

val_missing <- val_missing %>% 
  select(all_of(lot_col), site_id, company, city,
         ndc_11, all_of(mfr_name_col),
         ndma_ng_max_daily_dose, dmf_ng_max_daily_dose)

## E.  Optional: save output -----------------------------------------------
readr::write_csv(val_present,  file.path(OUT_DIR, "valisure_in_raw_redica.csv"))
readr::write_csv(val_missing,  file.path(OUT_DIR, "valisure_missing_from_raw_redica.csv"))
