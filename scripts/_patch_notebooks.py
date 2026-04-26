"""Patch 01_eda.ipynb (graph fixes) and 02_feature_analysis.ipynb (import fixes)."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# ── 01_eda.ipynb — cell 18 (boxplot overlap) ────────────────────────────────
CELL18 = [
    "key_num_cols = [\n",
    "    'Age', 'Annual_Income', 'Monthly_Inhand_Salary',\n",
    "    'Outstanding_Debt', 'Credit_Utilization_Ratio',\n",
    "    'Delay_from_due_date', 'Num_of_Delayed_Payment', 'Credit_History_Age'\n",
    "]\n",
    "key_num_cols = [c for c in key_num_cols if c in df.columns]\n",
    "\n",
    "# Coerce all key columns to numeric and parse Credit_History_Age string\n",
    "import re as _re\n",
    "plot_df = df.copy()\n",
    "if 'Credit_History_Age' in plot_df.columns and plot_df['Credit_History_Age'].dtype == object:\n",
    "    def _parse_cha(val):\n",
    "        if pd.isna(val):\n",
    "            return float('nan')\n",
    "        m = _re.search(r'(\\d+)\\s+Years?\\s+and\\s+(\\d+)\\s+Months?', str(val))\n",
    "        return int(m.group(1)) * 12 + int(m.group(2)) if m else float('nan')\n",
    "    plot_df['Credit_History_Age'] = plot_df['Credit_History_Age'].apply(_parse_cha)\n",
    "# Coerce remaining string columns to numeric\n",
    "for _col in key_num_cols:\n",
    "    if plot_df[_col].dtype == object:\n",
    "        plot_df[_col] = pd.to_numeric(\n",
    "            plot_df[_col].astype(str).str.replace(r'[^\\d.]', '', regex=True),\n",
    "            errors='coerce'\n",
    "        )\n",
    "\n",
    "fig, axes = plt.subplots(2, 4, figsize=(20, 9))\n",
    "axes = axes.flatten()\n",
    "order   = ['Poor', 'Standard', 'Good']\n",
    "palette = {'Poor': '#d9534f', 'Standard': '#f0ad4e', 'Good': '#5cb85c'}\n",
    "\n",
    "for i, col in enumerate(key_num_cols):\n",
    "    ax = axes[i]\n",
    "    sns.boxplot(\n",
    "        data=plot_df[plot_df['Credit_Score'].isin(order)],\n",
    "        x='Credit_Score', y=col, order=order,\n",
    "        palette=palette, ax=ax, showfliers=False\n",
    "    )\n",
    "    ax.set_title(col.replace('_', ' '), fontsize=11)\n",
    "    ax.set_xlabel('')\n",
    "    ax.yaxis.set_major_locator(mticker.MaxNLocator(5))\n",
    "    col_max = pd.to_numeric(plot_df[col], errors='coerce').dropna().max()\n",
    "    if isinstance(col_max, (int, float)) and col_max > 9999:\n",
    "        ax.yaxis.set_major_formatter(\n",
    "            mticker.FuncFormatter(lambda x, _: f'{x:,.0f}')\n",
    "        )\n",
    "    ax.tick_params(axis='y', labelsize=9)\n",
    "\n",
    "for j in range(i + 1, len(axes)):\n",
    "    axes[j].set_visible(False)\n",
    "\n",
    "plt.suptitle('Key Numeric Features vs Credit Score', fontsize=14, y=1.01)\n",
    "plt.tight_layout()\n",
    "plt.savefig(FIGURES_DIR / 'bivariate_numeric.png', dpi=150, bbox_inches='tight')\n",
    "plt.show()\n",
]

# ── 01_eda.ipynb — cell 19 (Credit_Mix garbage '_') ─────────────────────────
CELL19 = [
    "# Credit_Mix vs Credit Score — stacked bar (garbage '_' filtered)\n",
    "if 'Credit_Mix' in df.columns:\n",
    "    cm_data = df[df['Credit_Mix'].notna() & (df['Credit_Mix'] != '_')]\n",
    "    ct = pd.crosstab(cm_data['Credit_Mix'], cm_data['Credit_Score'],\n",
    "                     normalize='index') * 100\n",
    "    ct = ct.reindex(columns=['Poor', 'Standard', 'Good'], fill_value=0)\n",
    "    row_order = [r for r in ['Bad', 'Standard', 'Good'] if r in ct.index]\n",
    "    ct = ct.reindex(row_order)\n",
    "    fig, ax = plt.subplots(figsize=(8, 5))\n",
    "    ct.plot(kind='bar', stacked=True,\n",
    "            color=['#d9534f', '#f0ad4e', '#5cb85c'], ax=ax)\n",
    "    ax.set_title('Credit Mix vs Credit Score')\n",
    "    ax.set_ylabel('Percentage (%)')\n",
    "    ax.tick_params(axis='x', rotation=0)\n",
    "    ax.legend(title='Credit Score')\n",
    "    plt.tight_layout()\n",
    "    plt.savefig(FIGURES_DIR / 'credit_mix_vs_score.png', dpi=150, bbox_inches='tight')\n",
    "    plt.show()\n",
]

NB1 = ROOT / "notebooks" / "01_eda.ipynb"
with open(NB1, encoding="utf-8") as f:
    nb1 = json.load(f)

assert nb1["cells"][18]["cell_type"] == "code"
assert nb1["cells"][19]["cell_type"] == "code"
assert "key_num_cols" in "".join(nb1["cells"][18]["source"])
assert "Credit_Mix" in "".join(nb1["cells"][19]["source"])

nb1["cells"][18]["source"] = CELL18
nb1["cells"][18]["outputs"] = []
nb1["cells"][18]["execution_count"] = None

nb1["cells"][19]["source"] = CELL19
nb1["cells"][19]["outputs"] = []
nb1["cells"][19]["execution_count"] = None

with open(NB1, "w", encoding="utf-8") as f:
    json.dump(nb1, f, indent=1, ensure_ascii=False)
print("01_eda.ipynb — cells 18+19 patched")

# ── 02_feature_analysis.ipynb — cell 5 (correct imports + smart detection) ──
CELL5 = [
    "from src.features.build_features import build_features\n",
    "from src.features.selectors import FeatureSelector\n",
    "\n",
    "TARGET = 'Credit_Score'\n",
    "\n",
    "# Skip full pipeline if data is already OHE-preprocessed\n",
    "_is_processed = any(c.startswith('Occupation_') for c in df_train.columns)\n",
    "\n",
    "if not _is_processed:\n",
    "    from src.data.preprocessing import (\n",
    "        drop_pii_columns, clean_numeric_columns, parse_credit_history_age,\n",
    "        clean_categorical_columns, cap_outliers, encode_target,\n",
    "    )\n",
    "    from src.features.imputers import ImputerPipeline\n",
    "    from src.features.encoders import CategoricalEncoderPipeline\n",
    "\n",
    "    def _preprocess(df, cap=None):\n",
    "        df = drop_pii_columns(df)\n",
    "        df = clean_numeric_columns(df)\n",
    "        df = parse_credit_history_age(df)\n",
    "        df = clean_categorical_columns(df)\n",
    "        df, cap = cap_outliers(df, annual_income_cap=cap)\n",
    "        df = encode_target(df)\n",
    "        return df, cap\n",
    "\n",
    "    train_clean, cap = _preprocess(df_train.copy())\n",
    "    valid_clean, _   = _preprocess(df_valid.copy(), cap=cap)\n",
    "    imputer = ImputerPipeline()\n",
    "    enc = CategoricalEncoderPipeline()\n",
    "    train_enc = enc.fit_transform(imputer.fit_transform(train_clean))\n",
    "    valid_enc = enc.transform(imputer.transform(valid_clean))\n",
    "else:\n",
    "    print('Data already preprocessed - skipping pipeline.')\n",
    "    from src.features.imputers import ImputerPipeline\n",
    "    from src.features.encoders import CategoricalEncoderPipeline\n",
    "    imputer = ImputerPipeline()\n",
    "    enc = CategoricalEncoderPipeline()\n",
    "    train_enc = enc.fit_transform(imputer.fit_transform(df_train.copy()))\n",
    "    valid_enc = enc.transform(imputer.transform(df_valid.copy()))\n",
    "\n",
    "train_feat = build_features(train_enc)\n",
    "valid_feat = build_features(valid_enc)\n",
    "\n",
    "selector = FeatureSelector()\n",
    "X_train = selector.fit_transform(train_feat.drop(columns=[TARGET]))\n",
    "X_valid  = selector.transform(valid_feat.drop(columns=[TARGET]))\n",
    "y_train  = train_feat[TARGET]\n",
    "y_valid  = valid_feat[TARGET]\n",
    "\n",
    "print(f'X_train: {X_train.shape}, X_valid: {X_valid.shape}')\n",
]

CELL6_NOOP = [
    "# Preprocessing, imputation, encoding, and feature selection handled in the cell above.\n"
]

NB2 = ROOT / "notebooks" / "02_feature_analysis.ipynb"
with open(NB2, encoding="utf-8") as f:
    nb2 = json.load(f)

nb2["cells"][5]["source"] = CELL5
nb2["cells"][5]["outputs"] = []
nb2["cells"][5]["execution_count"] = None

nb2["cells"][6]["source"] = CELL6_NOOP
nb2["cells"][6]["outputs"] = []
nb2["cells"][6]["execution_count"] = None

with open(NB2, "w", encoding="utf-8") as f:
    json.dump(nb2, f, indent=1, ensure_ascii=False)
print("02_feature_analysis.ipynb — cells 5+6 patched")

# Verify
with open(NB2, encoding="utf-8") as f:
    nb2v = json.load(f)
src5 = "".join(nb2v["cells"][5]["source"])
assert "from src.features.build_features import build_features" in src5
assert "from src.features.selectors import FeatureSelector" in src5
assert "src.features.engineering" not in src5
assert "src.features.selection" not in src5
assert "_is_processed" in src5
print("Verification passed — all patches applied cleanly.")
