# export OPENAI_API_KEY=""

# === generate a log ===
LOG_DIR="../logs"
mkdir -p $LOG_DIR
LOG_FILE="$LOG_DIR/run_$(date +"%Y%m%d_%H%M%S").log"
exec > >(tee "$LOG_FILE") 2>&1

echo "log is saved to：$LOG_FILE"


GPT_VERSION="o4-mini"

# PAPER_NAME="Transformer"
# PDF_PATH="../examples/Transformer.pdf" # .pdf
# PDF_JSON_PATH="../examples/Transformer.json" # .json
# PDF_JSON_CLEANED_PATH="../examples/Transformer_cleaned.json" # _cleaned.json
# OUTPUT_DIR="../outputs/Transformer"
# OUTPUT_REPO_DIR="../outputs/Transformer_repo"

# PAPER_NAME="TSMOM_US"
# PDF_PATH="../examples/TSMOM_US.pdf"                    
# PDF_JSON_PATH="../examples/TSMOM_US.json"              
# PDF_JSON_CLEANED_PATH="../examples/TSMOM_US_cleaned.json"
# OUTPUT_DIR="../outputs/TSMOM_US"
# OUTPUT_REPO_DIR="../outputs/TSMOM_US_repo"

# PAPER_NAME="DRLPM"
# PDF_PATH="../examples/DRLPM.pdf"
# PDF_JSON_PATH="../examples/DRLPM.json"
# PDF_JSON_CLEANED_PATH="../examples/DRLPM_cleaned.json"
# OUTPUT_DIR="../outputs/DRLPM"
# OUTPUT_REPO_DIR="../outputs/DRLPM_repo"

PAPER_NAME="ADDPG"
PDF_PATH="../examples/ADDPG.pdf"
PDF_JSON_PATH="../examples/ADDPG.json"
PDF_JSON_CLEANED_PATH="../examples/ADDPG_cleaned.json"
OUTPUT_DIR="../outputs/ADDPG"
OUTPUT_REPO_DIR="../outputs/ADDPG_repo"


mkdir -p $OUTPUT_DIR
mkdir -p $OUTPUT_REPO_DIR

echo $PAPER_NAME

echo "------- Preprocess -------"

python ../codes/0_pdf_process.py \
    --input_json_path ${PDF_JSON_PATH} \
    --output_json_path ${PDF_JSON_CLEANED_PATH} \


echo "------- PaperCoder -------"

python ../codes/1_planning.py \
    --paper_name $PAPER_NAME \
    --gpt_version ${GPT_VERSION} \
    --pdf_json_path ${PDF_JSON_CLEANED_PATH} \
    --output_dir ${OUTPUT_DIR}


python ../codes/1.1_extract_config.py \
    --paper_name $PAPER_NAME \
    --output_dir ${OUTPUT_DIR}

cp -rp ${OUTPUT_DIR}/planning_config.yaml ${OUTPUT_REPO_DIR}/config.yaml

python ../codes/2_analyzing.py \
    --paper_name $PAPER_NAME \
    --gpt_version ${GPT_VERSION} \
    --pdf_json_path ${PDF_JSON_CLEANED_PATH} \
    --output_dir ${OUTPUT_DIR}

python ../codes/3_coding.py  \
    --paper_name $PAPER_NAME \
    --gpt_version ${GPT_VERSION} \
    --pdf_json_path ${PDF_JSON_CLEANED_PATH} \
    --output_dir ${OUTPUT_DIR} \
    --output_repo_dir ${OUTPUT_REPO_DIR} \
