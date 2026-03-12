## ============================================================================
## REALITY CHECK - Preprocessing Pipeline for Secret Dataset
## ============================================================================
## This script applies the same preprocessing steps developed on the training
## data to a new "secret" dataset for final model evaluation.
## NO exploratory analysis, NO model fitting - only data transformation.
## ============================================================================

## ============================================================================
## PRELIMINARIES
## ============================================================================

source("R/00_common.R")

##' Prepare raw LC data for modeling
##'
##' This function performs deterministic cleaning, type conversions, simple
##' imputation, and feature engineering on the raw LendingClub dataset. It is
##' intended to be safe for applying to new (holdout/production) data and
##' therefore avoids removals that would drop valid rows.
##'
##' @param LCdata A data.frame or tibble containing raw columns from the
##'        LendingClub dataset.
##' @return A tibble/data.frame with cleaned and engineered features ready
##'         for recipe preprocessing and modeling.
##' @examples
##' df <- readr::read_csv("data/LCdata.csv")
##' prepared <- prepare_data(df)
prepare_data <- function(LCdata) {
    ## ============================================================================
    ## STEP 0 — Setup & helpers
    ## ============================================================================
    get_mode <- function(x) {
    x <- x[!is.na(x)]
    if (length(x) == 0) {
        return(NA)
    }
    ux <- unique(x)
    ux[which.max(tabulate(match(x, ux)))]
    }

    ## ============================================================================
    ## STEP 1 — Manual feature removal (business rule)
    ## ============================================================================

    not_available_for_prediction <- c(
    "collection_recovery_fee", "installment", "funded_amnt", "funded_amnt_inv",
    "last_pymnt_amnt", "last_pymnt_d", "loan_status", "next_pymnt_d",
    "out_prncp", "out_prncp_inv", "pymnt_plan", "recoveries", "total_pymnt",
    "total_pymnt_inv", "total_rec_int", "total_rec_late_fee", "total_rec_prncp",
    "last_credit_pull_d", "issue_d"
    )

    drop_ident_free <- c(
    "id", "member_id", "url", "title", "desc", "emp_title", "zip_code"
    )

    constant_cols <- c(
    "policy_code"
    )

    drop_cols <- intersect(
    c(not_available_for_prediction, drop_ident_free, constant_cols),
    names(LCdata)
    )
    LCprep <- LCdata %>% dplyr::select(-all_of(drop_cols))

    ## ============================================================================
    ## STEP 2 — Light cleaning of obviously erroneous values
    ## ============================================================================

    if (any(c("dti", "dti_joint") %in% names(LCprep))) {
    LCprep <- LCprep %>%
        mutate(
        dti       = ifelse(!is.na(dti) & dti > 100, NA, dti),
        dti_joint = ifelse(!is.na(dti_joint) & dti_joint > 100, NA, dti_joint)
        )
    }

    if ("revol_util" %in% names(LCprep)) {
    LCprep <- LCprep %>%
        mutate(revol_util = ifelse(!is.na(revol_util) & revol_util > 100, NA, revol_util))
    }

    if (any(c("annual_inc", "annual_inc_joint") %in% names(LCprep))) {
    LCprep <- LCprep %>%
        mutate(
        annual_inc       = ifelse(!is.na(annual_inc) & annual_inc <= 0, NA, annual_inc),
        annual_inc_joint = ifelse(!is.na(annual_inc_joint) & annual_inc_joint <= 0, NA, annual_inc_joint)
        )
    }

    if ("term" %in% names(LCprep)) {
    LCprep <- LCprep %>%
        mutate(term = trimws(term))
    }

    ## ============================================================================
    ## STEP 3 — Type conversions / robust mappings
    ## ============================================================================

    # Term: Clean whitespace and map to standard levels
    if ("term" %in% names(LCprep)) {
    LCprep <- LCprep %>%
        mutate(
        term = trimws(term),
        term = case_when(
            term %in% c("36 months", "36", "36 months", " 36 months") ~ "36 months",
            term %in% c("60 months", "60", "60 months", " 60 months") ~ "60 months",
            TRUE ~ NA_character_
        ),
        term = factor(term, levels = c("36 months", "60 months"))
        )
    }

    # Application type: Standardize to Individual/Joint/Other
    if ("application_type" %in% names(LCprep)) {
    LCprep <- LCprep %>%
        mutate(
        application_type = case_when(
            str_detect(tolower(as.character(application_type)), "individual") ~ "Individual",
            str_detect(tolower(as.character(application_type)), "joint") ~ "Joint",
            is.na(application_type) ~ "Other",
            TRUE ~ "Other"
        ),
        application_type = factor(application_type, levels = c("Individual", "Joint", "Other"))
        )
    }

    # Initial list status: Normalize case and map to f/w/other
    if ("initial_list_status" %in% names(LCprep)) {
    LCprep <- LCprep %>%
        mutate(
        initial_list_status = tolower(trimws(as.character(initial_list_status))),
        initial_list_status = case_when(
            initial_list_status == "f" ~ "f",
            initial_list_status == "w" ~ "w",
            is.na(initial_list_status) ~ "other",
            TRUE ~ "other"
        ),
        initial_list_status = factor(initial_list_status, levels = c("f", "w", "other"))
        )
    }

    # Verification status: Map to standard categories with fallback
    if ("verification_status" %in% names(LCprep)) {
    LCprep <- LCprep %>%
        mutate(
        verification_status = case_when(
            as.character(verification_status) %in% c("Not Verified", "not verified") ~ "Not Verified",
            as.character(verification_status) %in% c("Verified", "verified") ~ "Verified",
            as.character(verification_status) %in% c("Source Verified", "source verified") ~ "Source Verified",
            is.na(verification_status) ~ "Other",
            TRUE ~ "Other"
        ),
        verification_status = factor(verification_status,
            levels = c("Not Verified", "Verified", "Source Verified", "Other")
        )
        )
    }

    # Address state: Map to regions with comprehensive coverage
    if ("addr_state" %in% names(LCprep)) {
    LCprep <- LCprep %>%
        mutate(
        addr_state = toupper(trimws(as.character(addr_state))),
        addr_region = case_when(
            addr_state %in% c("CT","ME","MA","NH","RI","VT","NJ","NY","PA") ~ "Northeast",
            addr_state %in% c("IL","IN","MI","OH","WI","IA","KS","MN","MO","NE","ND","SD") ~ "Midwest",
            addr_state %in% c("DE","FL","GA","MD","NC","SC","VA","DC","WV","AL","KY","MS","TN","AR","LA","OK","TX") ~ "South",
            addr_state %in% c("AZ","CO","ID","MT","NV","NM","UT","WY","AK","CA","HI","OR","WA") ~ "West",
            is.na(addr_state) ~ "Other",
            TRUE ~ "Other"
        ),
        addr_region = factor(addr_region, levels = c("Northeast","Midwest","South","West","Other"))
        ) %>%
        select(-any_of("addr_state"))
    }


    if ("emp_length" %in% names(LCprep)) {
    LCprep <- LCprep %>%
        mutate(
        emp_length_raw = trimws(emp_length),
        emp_length_raw = if_else(emp_length_raw %in% c("", "NA", "n/a"),
                                NA_character_, emp_length_raw)
        ) %>%
        mutate(
        .emp_length_num_temp = case_when(
            str_detect(emp_length_raw, "<") ~ 0.5,
            str_detect(emp_length_raw, "\\+") ~ 10,
            TRUE ~ parse_number(emp_length_raw)
        ),
        emp_length_factor = case_when(
            is.na(.emp_length_num_temp) ~ NA_character_,
            .emp_length_num_temp < 1 ~ "< 1 year",
            .emp_length_num_temp >= 10 ~ "10+ years",
            TRUE ~ paste0(.emp_length_num_temp, " years")
        ) %>%
            factor(
            levels = c("< 1 year", "1 years", "2 years", "3 years", "4 years",
                        "5 years", "6 years", "7 years", "8 years", "9 years", "10+ years"),
            ordered = TRUE
            ),
        emp_length_group = case_when(
            is.na(.emp_length_num_temp) ~ NA_character_,
            .emp_length_num_temp <= 1 ~ "low",
            .emp_length_num_temp <= 5 ~ "mid",
            .emp_length_num_temp > 5 ~ "high"
        ) %>%
            factor(levels = c("low", "mid", "high"), ordered = TRUE)
        ) %>%
        select(-any_of(c(".emp_length_num_temp", "emp_length")))
    }

    ## ============================================================================
    ## STEP 4 — Co-borrower handling 
    ## ============================================================================

    if (!"annual_inc" %in% names(LCprep)) LCprep$annual_inc <- NA_real_
    if (!"annual_inc_joint" %in% names(LCprep)) LCprep$annual_inc_joint <- NA_real_
    if (!"dti" %in% names(LCprep)) LCprep$dti <- NA_real_
    if (!"dti_joint" %in% names(LCprep)) LCprep$dti_joint <- NA_real_
    if (!"verification_status_joint" %in% names(LCprep)) LCprep$verification_status_joint <- NA_character_

    LCprep <- LCprep %>%
    mutate(
        is_joint_app = case_when(
        !is.na(application_type) & str_detect(as.character(application_type), regex("joint", ignore_case = TRUE)) ~ TRUE,
        !is.na(application_type) ~ FALSE,
        TRUE ~ NA
        ),
        verification_status_joint = case_when(
        is_joint_app == TRUE & !is.na(verification_status_joint) ~ as.character(verification_status_joint),
        is_joint_app == TRUE &  is.na(verification_status_joint) ~ "Missing",
        TRUE ~ "None"
        ),
        verification_status_joint = factor(
        verification_status_joint,
        levels = c("None", "Not Verified", "Verified", "Source Verified", "Missing")
        ),
        annual_inc_combined = case_when(
        is_joint_app == TRUE & !is.na(annual_inc_joint) ~ annual_inc_joint,
        TRUE ~ annual_inc
        ),
        dti_combined = dplyr::coalesce(dti_joint, dti)
    ) %>%
    mutate(
        .main_verified  = as.character(verification_status) %in% c("Verified", "Source Verified"),
        .joint_verified = as.character(verification_status_joint) %in% c("Verified", "Source Verified"),
        verification_status_combined = case_when(
        .main_verified & .joint_verified  ~ "Both Verified",
        .main_verified & !.joint_verified ~ "Main Only",
        !.main_verified & .joint_verified ~ "Joint Only",
        TRUE                              ~ "None Verified"
        ),
        verification_status_combined = factor(
        verification_status_combined,
        levels = c("None Verified", "Main Only", "Joint Only", "Both Verified")
        )
    ) %>%
    select(-any_of(c(".main_verified", ".joint_verified")))

    ## ============================================================================
    ## STEP 5 — Simple imputation
    ## ============================================================================

    num_cols <- names(LCprep)[sapply(LCprep, is.numeric)]
    fac_cols <- names(LCprep)[sapply(LCprep, is.factor)]

    for (cn in num_cols) {
    med <- median(LCprep[[cn]], na.rm = TRUE)
    if (is.finite(med)) {
        LCprep[[cn]][is.na(LCprep[[cn]])] <- med
    }
    }

    for (cn in fac_cols) {
    if (all(is.na(LCprep[[cn]]))) {
        LCprep[[cn]] <- factor(LCprep[[cn]], levels = c(levels(LCprep[[cn]]), "Missing"))
        LCprep[[cn]][is.na(LCprep[[cn]])] <- "Missing"
    } else {
        m <- get_mode(LCprep[[cn]])
        LCprep[[cn]][is.na(LCprep[[cn]])] <- m
        LCprep[[cn]] <- droplevels(LCprep[[cn]])
    }
    }

    ## ============================================================================
    ## STEP 6 — Feature engineering
    ## ============================================================================

    ## 6a) Income per loan + log transformation (NO filtering for secret dataset)

    LCprep <- LCprep %>%
    mutate(
        income_per_loan     = annual_inc_combined / loan_amnt,
        income_per_loan_log = log(income_per_loan + 1)
    )

    # NOTE: For the secret dataset, we do NOT filter outliers
    # We must predict all applications, even those with extreme values

    ## 6b) Issue date (EXCLUDED - not available for new applications)

    ## 6c) Credit history years (EXCLUDED - unsuitable for holdout set)

    ## 6d) Home ownership - collapse rare categories and handle NAs

    if ("home_ownership" %in% names(LCprep)) {
    LCprep <- LCprep %>%
        mutate(
        home_ownership = case_when(
            as.character(home_ownership) %in% c("ANY", "NONE") ~ "OTHER",
            is.na(home_ownership) ~ NA_character_,
            TRUE ~ as.character(home_ownership)
        ),
        home_ownership = forcats::fct_na_value_to_level(
            factor(home_ownership),
            "Unknown"
        )
        )
    }

    ## 6e) Earliest credit line → Numeric month index

    LCprep <- LCprep %>%
    mutate(
        .earliest_cr_line_date = as.Date(
        paste0("01-", earliest_cr_line),
        format = "%d-%b-%Y"
        ),
        earliest_cr_line_index = ifelse(
        is.na(.earliest_cr_line_date),
        NA_integer_,
        as.integer(format(.earliest_cr_line_date, "%Y")) * 12 +
            as.integer(format(.earliest_cr_line_date, "%m"))
        )
    )

    med_idx <- median(LCprep$earliest_cr_line_index, na.rm = TRUE)

    LCprep <- LCprep %>%
    mutate(
        earliest_cr_line_index = ifelse(
        is.na(earliest_cr_line_index),
        med_idx,
        earliest_cr_line_index
        )
    ) %>%
    select(-any_of(c("earliest_cr_line", ".earliest_cr_line_date")))

    ## 6f) Loan purpose consolidation

    if ("purpose" %in% names(LCprep)) {
    LCprep <- LCprep %>%
        mutate(
        purpose_grouped = case_when(
            as.character(purpose) == "debt_consolidation" ~ "debt_consolidation",
            as.character(purpose) == "credit_card" ~ "credit_card",
            as.character(purpose) == "home_improvement" ~ "home_improvement",
            as.character(purpose) == "major_purchase" ~ "major_purchase",
            as.character(purpose) == "small_business" ~ "small_business",
            as.character(purpose) %in% c("car", "medical", "moving", "vacation",
                                        "wedding", "house", "renewable_energy",
                                        "educational", "other") ~ "other_small",
            is.na(purpose) ~ "other_small",  # Map NA to fallback
            TRUE ~ "other_small"  # Map any unseen category to fallback
        ),
        purpose_grouped = factor(purpose_grouped,
                                levels = c("debt_consolidation", "credit_card",
                                            "home_improvement", "major_purchase",
                                            "small_business", "other_small"))
        ) %>%
        select(-any_of("purpose"))
    }

    ## 6g) Credit score feature (vectorized, based on combined fields)

    ##' Compute a heuristic credit score
    ##'
    ##' A small helper to compute a composite score from a handful of inputs.
    ##' This is vectorized and returns a numeric vector the same length as inputs.
    ##'
    ##' @param dti_combined Numeric debt-to-income ratio.
    ##' @param income_per_loan Numeric income-per-loan feature.
    ##' @param annual_inc_combined Numeric annual income (possibly combined).
    ##' @param revol_util Numeric revolving utilization percentage (0-100).
    ##' @param inq Integer number of inquiries in last 6 months.
    ##' @param emp_length_raw Character or numeric employment length raw field.
    ##' @return Numeric vector with heuristic credit score (higher == better).
    ##' @examples
    ##' credit_score(10, 2.5, 75000, 20, 1, "2 years")
    credit_score <- function(dti_combined, income_per_loan, annual_inc_combined,
                                revol_util, inq, emp_length_raw) {
    emp_len <- dplyr::case_when(
        is.na(emp_length_raw) ~ NA_real_,
        str_detect(emp_length_raw, "<") ~ 0.5,
        str_detect(emp_length_raw, "\\+") ~ 10,
        TRUE ~ parse_number(emp_length_raw)
    )
    
    dti_score <- dplyr::case_when(
        dti_combined < 10 ~ 20,
        dti_combined < 20 ~ 15,
        dti_combined < 30 ~ 10,
        dti_combined < 40 ~ 5,
        TRUE ~ 0
    )

    ipl_score <- dplyr::case_when(
        income_per_loan > 10 ~ 20,
        income_per_loan > 5 ~ 15,
        income_per_loan > 3 ~ 10,
        income_per_loan > 1 ~ 5,
        TRUE ~ 0
    )

    inc_score <- dplyr::case_when(
        annual_inc_combined > 120000 ~ 15,
        annual_inc_combined > 70000 ~ 12,
        annual_inc_combined > 40000 ~ 8,
        annual_inc_combined > 20000 ~ 4,
        TRUE ~ 0
    )

    util_score <- dplyr::case_when(
        revol_util < 20 ~ 20,
        revol_util < 40 ~ 15,
        revol_util < 60 ~ 10,
        revol_util < 80 ~ 5,
        TRUE ~ 0
    )

    inq_score <- dplyr::case_when(
        inq == 0 ~ 10,
        inq <= 2 ~ 7,
        inq <= 4 ~ 4,
        TRUE ~ 0
    )

    emp_score <- dplyr::case_when(
        is.na(emp_len) ~ 0,
        emp_len >= 10 ~ 15,
        emp_len >= 5 ~ 12,
        emp_len >= 2 ~ 8,
        emp_len >= 1 ~ 4,
        TRUE ~ 0
    )

    dti_score + ipl_score + inc_score + util_score + inq_score + emp_score
    }

    LCprep <- LCprep %>%
    mutate(
        credit_score = credit_score(
        dti_combined        = dti_combined,
        income_per_loan     = income_per_loan,
        annual_inc_combined = annual_inc_combined,
        revol_util          = revol_util,
        inq                 = inq_last_6mths,
        emp_length_raw      = emp_length_raw
        )
    )

    ## 6h) Revolving utilization × Loan amount interaction

    LCprep <- LCprep %>%
    mutate(
        revol_util_prop = revol_util / 100,
        util_x_loan     = revol_util_prop * loan_amnt
    )

    ## 6i) Additional risk features and interactions

    LCprep <- LCprep %>%
    mutate(
        revol_bal_limit_ratio = revol_bal / pmax(total_rev_hi_lim, 1),
        revol_bal_limit_ratio_log = log(revol_bal_limit_ratio + 1),
        revol_bal_ratio_bucket = cut(
        revol_bal_limit_ratio,
        breaks = c(-Inf, 0, 0.2, 0.5, 0.8, Inf),
        labels = c("none", "very_low", "low", "medium", "high")
        )
    )

    LCprep <- LCprep %>%
    mutate(
        interaction_dti_revol = dti_combined * revol_util,
        interaction_dti_revol_log = log(interaction_dti_revol + 1)
    )

    ## 6j) Drop original variables replaced by engineered features

    original_vars_to_drop <- c(
    "emp_length", "emp_length_factor", "emp_length_raw",
    "annual_inc", "annual_inc_joint",
    "dti", "dti_joint", "is_joint_app",
    "verification_status", "verification_status_joint",
    "earliest_cr_line",
    "revol_util",
    "revol_bal_limit_ratio",
    "income_per_loan",
    "interaction_dti_revol"
    )

    LCprep <- LCprep %>%
    select(-any_of(original_vars_to_drop))

        LCprep <- LCprep %>% mutate(across(where(is.character), as.factor))

    return(LCprep)
}

##' Convert preprocessed data and a recipe to an XGBoost DMatrix
##'
##' This function `bake()`s the provided `recipe` on `processed_data`, converts
##' predictors to a sparse matrix and returns an `xgb.DMatrix` with labels.
##'
##' @param processed_data A data.frame/tibble produced by `prepare_data()`.
##' @param recipe A prepped recipe object (result of `prep()`).
##' @param target_var Name of the target column (default: `TARGET_VAR`).
##' @return An `xgb.DMatrix` usable by `xgb.train()`.
##' @examples
##' rec <- train_recipe(train_df, seed = 42)
##' dmat <- convert_to_xgb_matrix(train_df, rec, "int_rate")
convert_to_xgb_matrix <- function(processed_data, recipe, target_var = TARGET_VAR) {
    baked_data <- bake(recipe, new_data = processed_data)
    x_data <- baked_data %>% dplyr::select(-all_of(target_var))
    y_data <- baked_data[[target_var]]
    matrix <- Matrix::Matrix(data.matrix(x_data), sparse = TRUE)

    return(xgb.DMatrix(data = matrix, label = y_data))
}

##' Convenience: prepare raw data and convert to XGBoost DMatrix
##'
##' Takes raw `LCdata`, applies `prepare_data()` and then returns an
##' `xgb.DMatrix` by calling `convert_to_xgb_matrix()`.
##'
##' @param LCdata Raw data.frame/tibble.
##' @param recipe A prepped recipe object.
##' @param target_var Target column name.
##' @return An `xgb.DMatrix`.
##' @examples
##' rec <- train_recipe(train_df, seed = 42)
##' dmat <- raw_data_to_xgb_matrix(raw_df, rec)
raw_data_to_xgb_matrix <- function(LCdata, recipe, target_var = TARGET_VAR) {
    prep_data <- prepare_data(LCdata)
    xgb_matrix <- convert_to_xgb_matrix(prep_data, recipe, target_var)

    return(xgb_matrix)
}

# 1. Get processed data
#LCprep <- readr::read_csv("LC_clean_baseline.csv", show_col_types = FALSE)
# 2. Convert character columns back to factors (CSV loses factor metadata)
#LCprep <- LCprep %>% mutate(across(where(is.character), as.factor))
# 3. Split into train/test
# 4. Train recipe on training set
# rec_all <- recipe(int_rate ~ ., data = train_full_xgb) %>%
#   step_unknown(all_nominal_predictors()) %>%
#   step_novel(all_nominal_predictors()) %>%
#   step_impute_mode(all_nominal_predictors()) %>%
#   step_impute_median(all_numeric_predictors()) %>%
#   step_dummy(all_nominal_predictors(), one_hot = TRUE) %>%
#   step_zv(all_predictors())
# prep_rec_all <- prep(rec_all, training = train_full_xgb, retain = TRUE)
# 5. Apply recipe to training and test sets
# train_baked_all <- bake(prep_rec_all, new_data = train_full_xgb)
# test_baked_all  <- bake(prep_rec_all, new_data = test_full_xgb)