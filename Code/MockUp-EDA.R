# ==============================================================================
# ================================ Description =================================
# ==============================================================================
# Author: Amirreza Sahebi
# Last modified: 11.17.2024
# Input: DOD Sample dataset
# Output: Dataframe with geocoded locations and visualizations
# Purpose: Analyze and visualize Tier 1 medicines, including geolocation data


# ==============================================================================
# ============================= Required Libraries =============================
# ==============================================================================
library(readxl)
library(dplyr)
library(ggplot2)
library(stringr)
library(viridis)
library(tidyr)
library(ggrepel)

# ==============================================================================
# ================================= Read Data  =================================
# ==============================================================================
pathRead <- "G:/My Drive/North Carolina State University/Project - Drug Shortage/Data/"
pathWrite <- "G:/My Drive/North Carolina State University/Project - Drug Shortage/Output/"

# Load the datasets
dfDODSample <- read_excel(paste0(pathRead, "Prj_initial Sample.xlsx"), sheet = "DOD Sample")
dfOralNet <- read_excel(paste0(pathRead, "Oral_Net.xlsx"))

# View the first few rows to understand structure
glimpse(dfDODSample)

# ==============================================================================
# ============================== Data Processing ============================== #
# ==============================================================================

# Calculate HHI (Herfindahl-Hirschman Index)
dfDODSample <- dfDODSample %>%
  group_by(substancename_part, dosageformname) %>%
  mutate(
    market_share = units_reimbursed / sum(units_reimbursed, na.rm = TRUE),
    HHI = sum(market_share^2, na.rm = TRUE)
  ) %>%
  ungroup()

# Define Tier 1 medicines and Shortage Medicines
Tier1Medicines <- c("ALBUTEROL SULFATE", "AMIODARONE HYDROCHLORIDE", "DEXAMETHASONE",
                    "FUROSEMIDE", "HALOPERIDOL", "HYDRALAZINE HYDROCHLORIDE", 
                    "LORAZEPAM", "METHYLPREDNISOLONE", "METRONIDAZOLE",
                    "MIDAZOLAM HYDROCHLORIDE", "MORPHINE SULFATE", "ONDANSETRON",
                    "POTASSIUM CHLORIDE", "TACROLIMUS", "VANCOMYCIN HYDROCHLORIDE")

ShortageMedicines <- c("ALBUTEROL SULFATE", "ATROPINE", "DEXAMETHASONE", "EPINEPHRINE", 
                       "FENTANYL", "FUROSEMIDE", "HEPARIN", "LIDOCAINE + EPINEPHRINE", 
                       "LORAZEPAM", "METHYLPREDNISOLONE", "METRONIDAZOLE", "MIDAZOLAM", 
                       "MORPHINE", "POTASSIUM CHLORIDE")

# Filter for Tier 1 medicines
dfDODSample <- dfDODSample %>%
  filter(substancename_part %in% Tier1Medicines) %>%
  mutate(shortage_status = ifelse(substancename_part %in% ShortageMedicines,1,0),
         substancename_part = str_to_title(substancename_part))

# Join with manufacturer data
dfDODSample <- dfDODSample %>%
  left_join(dfOralNet, by = c("PRODUCTNDC" = "Productndc"))

# Extract location data from the "Site Display Name" column
dfDODSample <- dfDODSample %>%
  mutate(
    # Extract the part inside brackets
    location_info = str_extract(`Site Display Name`, "\\[.*\\]"),
    
    # Remove brackets and split by " / "
    location_info = str_replace_all(location_info, "\\[|\\]", ""), # Remove [ and ]
    
    # Split into City and Country
    City = str_trim(str_extract(location_info, "^[^/]+")),          # Extract before "/"
    Country = str_trim(str_extract(location_info, "(?<=/).*")),    # Extract after "/"
    
    # Capitalize City and Country for consistency
    City = str_to_title(City),
    Country = str_to_title(Country)
  ) %>%
  select(-location_info)  # Drop intermediate column if not needed

dfDODSample$`Applicant Full Name (ORANGE_BOOK)` <- str_to_title(dfDODSample$`Applicant Full Name (ORANGE_BOOK)`)

# Save the merged data
write.csv(dfDODSample, paste0(pathWrite, "Merged_DOD_Sample.csv"), row.names = FALSE)

# ==============================================================================
# =============================== Aggregated Data ==============================
# ==============================================================================
dfDODSampleAgg <- dfDODSample %>%
  group_by(substancename_part, Country) %>%
  summarise(
    mean_total_score = mean(`Total Score`, na.rm = T),
    mean_L_10_Y_score = mean(mean_L_10_Y_score, na.rm = TRUE),
    total_market_size = sum(mkt_size, na.rm = TRUE),
    avg_hhi = mean(HHI, na.rm = TRUE),
    n_mfr = sum(n_mfr, na.rm = TRUE),
    shortage_status = ifelse(unique(substancename_part) %in% ShortageMedicines, "Yes", "No"),
    .groups = "drop"
  ) %>%
  mutate(substancename_part = str_to_title(substancename_part)) # Capitalize medicine names

# Save aggregated data
write.csv(dfDODSampleAgg, paste0(pathWrite, "Aggregated_DOD_Sample.csv"), row.names = FALSE)

# ==============================================================================
# =================================== Plots ====================================
# ==============================================================================

# 1. Mean 10-Year Scores by Drug and Country
score_plot <- ggplot(dfDODSampleAgg, aes(x = substancename_part, y = mean_total_score, fill = Country)) +
  geom_bar(stat = "identity", position = "dodge", color = "black") +
  scale_fill_viridis_d(option = "plasma") +
  theme_minimal(base_family = "mono") +
  labs(
    title = "Mean Total Scores by Drug and Country",
    x = "Drug",
    y = "Mean Total Score",
    fill = "Country"
  ) +
  theme_light() +  # Use theme with gridlines
  theme(
    strip.text.x = element_text(size = 12, color = "black", face = "bold", family = "mono"),
    strip.text.y = element_text(size = 12, color = "black", face = "bold", family = "mono"),
    legend.title = element_text(family = "mono", size = 10),
    legend.position = "bottom",
    legend.text = element_text(family = "mono", size = 10),
    axis.title.y = element_text(family = "mono", size = 14, face = "bold"),
    axis.title.x = element_text(family = "mono", size = 14, face = "bold", hjust = 0.5),
    axis.text.x = element_text(family = "mono", size = 10, face = "bold", angle = 15, hjust = 1),  # Adjust angle and alignment
    axis.text.y = element_text(family = "mono", size = 12, face = "bold"),
    plot.title = element_text(family = "mono", size = 16, face = "bold", hjust = 0.5),
    plot.caption = element_text(family = "mono", size = 12, hjust = 0, vjust = 1),
    plot.caption.position = "plot",
    plot.margin = margin(t = 10, r = 10, b = 30, l = 10)  # Add more space at the bottom
  )
ggsave(paste0(pathWrite, "Mean_L_10_Y_Scores_by_Country.png"), plot = score_plot, width = 14, height = 8, bg = "white")

# 2. Bubble Plot: Market Size, HHI, and Number of Manufacturers
dfTmp <- dfDODSampleAgg %>%
  group_by(substancename_part) %>%
  summarise(
    total_market_size = sum(total_market_size, na.rm = TRUE),
    avg_hhi = mean(avg_hhi, na.rm = TRUE),
    n_mfr = sum(n_mfr, na.rm = TRUE),
    shortage_status = ifelse(any(shortage_status == "Yes"), "Yes", "No")
  )

# Bubble Plot using the aggregated dfTmp
plot_bubble <- ggplot(dfTmp, aes(
  x = total_market_size,
  y = avg_hhi,
  size = n_mfr,
  color = shortage_status,
  label = substancename_part
)) +
  geom_point(alpha = 0.7) +
  geom_text_repel(size = 4, max.overlaps = 10, box.padding = 0.5, point.padding = 0.5) +
  scale_color_manual(values = c("Yes" = "red", "No" = "steelblue")) +
  scale_size_continuous(range = c(3, 15)) +
  scale_x_log10() +
  theme_minimal(base_family = "mono") +
  labs(
    title = "Market Size, HHI, and Manufacturers by Medicine",
    x = "Log-Scaled Total Market Size (Units)",
    y = "Average HHI",
    size = "Number of Manufacturers",
    color = "In Shortage"
  ) +
  theme_light() +  # Use theme with gridlines
  theme(
    strip.text.x = element_text(size = 12, color = "black", face = "bold", family = "mono"),
    strip.text.y = element_text(size = 12, color = "black", face = "bold", family = "mono"),
    legend.title = element_text(family = "mono", size = 10),
    legend.position = "bottom",
    legend.text = element_text(family = "mono", size = 10),
    axis.title.y = element_text(family = "mono", size = 14, face = "bold"),
    axis.title.x = element_text(family = "mono", size = 14, face = "bold", hjust = 0.5),
    axis.text.x = element_text(family = "mono", size = 10, face = "bold", angle = 15, hjust = 1),  # Adjust angle and alignment
    axis.text.y = element_text(family = "mono", size = 12, face = "bold"),
    plot.title = element_text(family = "mono", size = 16, face = "bold", hjust = 0.5),
    plot.caption = element_text(family = "mono", size = 12, hjust = 0, vjust = 1),
    plot.caption.position = "plot",
    plot.margin = margin(t = 10, r = 10, b = 30, l = 10)  # Add more space at the bottom
  )
# Save the plot
ggsave(paste0(pathWrite, "Market_Size_HHI_Manufacturers_Bubble.png"), plot = plot_bubble, width = 16, height = 10, bg = "white")
