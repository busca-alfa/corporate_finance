import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from pathlib import Path
import json
import re
import io
import zipfile
from datetime import datetime

DATA_DIR = Path("data_empresas")
DATA_DIR.mkdir(exist_ok=True)

def slugify(nome: str) -> str:
    s = (nome or "").strip().lower()
    s = re.sub(r"[^a-z0-9]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s or "empresa"

def empresa_path(empresa_id: str) -> Path:
    return DATA_DIR / f"{empresa_id}.json"

def df_to_records(df: pd.DataFrame) -> list:
    if df is None:
        return []
    return df.to_dict(orient="records")

def records_to_df(records: list) -> pd.DataFrame:
    if not records:
        return pd.DataFrame()
    return pd.DataFrame.from_records(records)

def salvar_empresa(empresa_id: str, payload: dict) -> None:
    path = empresa_path(empresa_id)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

def carregar_empresa(empresa_id: str) -> dict | None:
    path = empresa_path(empresa_id)
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)

def listar_empresas() -> list[tuple[str, str]]:
    out = []
    for p in sorted(DATA_DIR.glob("*.json")):
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
            out.append((p.stem, data.get("empresa_nome", p.stem)))
        except Exception:
            out.append((p.stem, p.stem))
    return out

def restaurar_para_session_state(data: dict):
    """Restaura DFs no session_state a partir do JSON."""
    if not data:
        return
    st.session_state["empresa_id"] = data.get("empresa_id", "")
    st.session_state["empresa_nome"] = data.get("empresa_nome", "")

    st.session_state["dre_raw"] = records_to_df(data.get("dre_raw", []))
    ov_dre = records_to_df(data.get("dre_override", []))
    st.session_state["dre_override"] = ov_dre.set_index("Conta") if not ov_dre.empty and "Conta" in ov_dre.columns else st.session_state.get("dre_override")

    st.session_state["bp_raw"] = records_to_df(data.get("bp_raw", []))
    ov_bp = records_to_df(data.get("bp_override", []))
    st.session_state["bp_override"] = ov_bp.set_index("Conta") if not ov_bp.empty and "Conta" in ov_bp.columns else st.session_state.get("bp_override")

def coletar_payload_do_session_state(empresa_id: str, empresa_nome: str) -> dict:
    """Monta o JSON persistível com base no que está no app."""
    dre_override_df = st.session_state.get("dre_override")
    bp_override_df = st.session_state.get("bp_override")

    payload = {
        "empresa_id": empresa_id,
        "empresa_nome": empresa_nome,
        "salvo_em": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "dre_raw": df_to_records(st.session_state.get("dre_raw")),
        "dre_override": df_to_records(dre_override_df.reset_index()) if isinstance(dre_override_df, pd.DataFrame) else [],
        "bp_raw": df_to_records(st.session_state.get("bp_raw")),
        "bp_override": df_to_records(bp_override_df.reset_index()) if isinstance(bp_override_df, pd.DataFrame) else [],
    }
    return payload

def empresa_existe(empresa_id: str) -> bool:
    return empresa_path(empresa_id).exists()

def gerar_excel_bytes(dfs: dict[str, pd.DataFrame]) -> bytes:
    """
    Gera um XLSX em memória com várias abas.
    Requer openpyxl instalado.
    """
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        for nome, df in dfs.items():
            if df is None:
                continue
            if isinstance(df, pd.DataFrame) and not df.empty:
                df.to_excel(writer, sheet_name=nome[:31], index=True)
            else:
                pd.DataFrame().to_excel(writer, sheet_name=nome[:31], index=False)
    output.seek(0)
    return output.read()

def gerar_zip_empresa(empresa_id: str, data: dict) -> bytes:
    """ZIP contendo JSON e um XLSX com as tabelas principais."""
    zip_buf = io.BytesIO()
    with zipfile.ZipFile(zip_buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        # JSON
        zf.writestr(f"{empresa_id}.json", json.dumps(data, ensure_ascii=False, indent=2))

        # Excel
        dre_raw = records_to_df(data.get("dre_raw", []))
        dre_override = records_to_df(data.get("dre_override", []))
        bp_raw = records_to_df(data.get("bp_raw", []))
        bp_override = records_to_df(data.get("bp_override", []))

        # Ajusta index (se vier com coluna Conta no override)
        if "Conta" in dre_override.columns:
            dre_override = dre_override.set_index("Conta")
        if "Conta" in bp_override.columns:
            bp_override = bp_override.set_index("Conta")

        xlsx = gerar_excel_bytes({
            "DRE_raw": dre_raw.set_index("Conta") if "Conta" in dre_raw.columns else dre_raw,
            "DRE_override": dre_override,
            "BP_raw": bp_raw.set_index("Conta") if "Conta" in bp_raw.columns else bp_raw,
            "BP_override": bp_override,
        })
        zf.writestr(f"{empresa_id}.xlsx", xlsx)

    zip_buf.seek(0)
    return zip_buf.read()

st.set_page_config(
    page_title="Análise Econômico-Financeira",
    layout="wide"
)

st.title("📊 Análise Econômico-Financeira da Empresa")
st.caption("Preencha os dados do mais antigo para o mais recente")

st.sidebar.markdown("## 🗂️ Gestão de Empresas")

empresas = listar_empresas()
mapa_nome = {eid: nome for eid, nome in empresas}
ids = [eid for eid, _ in empresas]

# Seleção atual (persistida)
eid_atual = st.session_state.get("empresa_id", "")
nome_atual = st.session_state.get("empresa_nome", "")

# Se não tem empresa ainda, sugere nova
if not ids:
    st.sidebar.info("Nenhuma empresa cadastrada ainda. Crie uma nova abaixo.")

opcoes = ["— Nova empresa —"] + [f"{mapa_nome[eid]}  ({eid})" for eid in ids]
sel = st.sidebar.selectbox("Empresa", opcoes, index=0 if not eid_atual else (1 + ids.index(eid_atual) if eid_atual in ids else 0))

# Campos de Nova empresa
novo_nome = ""
novo_id = ""
if sel == "— Nova empresa —":
    novo_nome = st.sidebar.text_input("Nome da empresa", placeholder="Ex.: ACME S.A.")
    novo_id = slugify(novo_nome) if novo_nome.strip() else ""
else:
    novo_id = sel.split("(")[-1].replace(")", "").strip()
    novo_nome = mapa_nome.get(novo_id, novo_id)

st.sidebar.caption("Dica: salve frequentemente. Você poderá duplicar e renomear depois.")

c1, c2 = st.sidebar.columns(2)
btn_carregar = c1.button("📥 Carregar", use_container_width=True)
btn_salvar = c2.button("💾 Salvar", use_container_width=True)

# Ações principais
if btn_carregar:
    if not novo_id:
        st.sidebar.warning("Informe/Selecione uma empresa.")
    else:
        data = carregar_empresa(novo_id)
        if not data:
            st.sidebar.info("Empresa ainda sem dados salvos. Preencha e clique em Salvar.")
            st.session_state["empresa_id"] = novo_id
            st.session_state["empresa_nome"] = novo_nome
        else:
            restaurar_para_session_state(data)
            st.sidebar.success(f"Carregado: {novo_nome} ({novo_id})")
            st.rerun()

if btn_salvar:
    if not novo_id:
        st.sidebar.warning("Informe/Selecione uma empresa antes de salvar.")
    else:
        st.session_state["empresa_id"] = novo_id
        st.session_state["empresa_nome"] = novo_nome
        payload = coletar_payload_do_session_state(novo_id, novo_nome)
        salvar_empresa(novo_id, payload)
        st.sidebar.success(f"Salvo: {novo_nome} ({novo_id})")
        st.rerun()

st.sidebar.divider()

# ====== Duplicar ======
st.sidebar.markdown("### 📄 Duplicar")
dup_nome = st.sidebar.text_input("Novo nome (duplicação)", placeholder="Ex.: ACME S.A. (cenário 2)")
btn_duplicar = st.sidebar.button("Duplicar empresa", use_container_width=True)

if btn_duplicar:
    if not novo_id or not empresa_existe(novo_id):
        st.sidebar.warning("Selecione uma empresa existente para duplicar.")
    elif not dup_nome.strip():
        st.sidebar.warning("Informe o novo nome.")
    else:
        data = carregar_empresa(novo_id)
        new_id = slugify(dup_nome)
        # evitar overwrite acidental
        if empresa_existe(new_id):
            st.sidebar.warning("Já existe uma empresa com esse ID. Ajuste o nome.")
        else:
            data["empresa_id"] = new_id
            data["empresa_nome"] = dup_nome.strip()
            salvar_empresa(new_id, data)
            st.sidebar.success(f"Duplicado para: {dup_nome} ({new_id})")
            st.rerun()

st.sidebar.divider()

# ====== Renomear ======
st.sidebar.markdown("### ✏️ Renomear")
rename_nome = st.sidebar.text_input("Novo nome (renomear)", placeholder="Ex.: ACME S.A. — Consolidado")
btn_renomear = st.sidebar.button("Renomear (mantém ID)", use_container_width=True)

if btn_renomear:
    if not novo_id or not empresa_existe(novo_id):
        st.sidebar.warning("Selecione uma empresa existente.")
    elif not rename_nome.strip():
        st.sidebar.warning("Informe o novo nome.")
    else:
        data = carregar_empresa(novo_id)
        data["empresa_nome"] = rename_nome.strip()
        salvar_empresa(novo_id, data)
        # se for a empresa atual, atualiza session_state
        if st.session_state.get("empresa_id") == novo_id:
            st.session_state["empresa_nome"] = rename_nome.strip()
        st.sidebar.success("Renomeado com sucesso.")
        st.rerun()

st.sidebar.divider()

# ====== Exportar ======
st.sidebar.markdown("### 📦 Exportar")
btn_exportar = st.sidebar.button("Gerar ZIP (JSON + Excel)", use_container_width=True)

if btn_exportar:
    if not novo_id or not empresa_existe(novo_id):
        st.sidebar.warning("Selecione uma empresa existente.")
    else:
        data = carregar_empresa(novo_id)
        zip_bytes = gerar_zip_empresa(novo_id, data)
        st.sidebar.download_button(
            label="⬇️ Baixar ZIP",
            data=zip_bytes,
            file_name=f"{novo_id}_export.zip",
            mime="application/zip",
            use_container_width=True
        )

st.sidebar.divider()

# ====== Deletar ======
st.sidebar.markdown("### 🗑️ Deletar")
conf_del = st.sidebar.checkbox("Confirmo que quero deletar esta empresa (irreversível)")
btn_deletar = st.sidebar.button("Deletar empresa", use_container_width=True)

if btn_deletar:
    if not novo_id or not empresa_existe(novo_id):
        st.sidebar.warning("Selecione uma empresa existente.")
    elif not conf_del:
        st.sidebar.warning("Marque a confirmação para deletar.")
    else:
        empresa_path(novo_id).unlink(missing_ok=True)
        # se deletou a atual, limpa seleção
        if st.session_state.get("empresa_id") == novo_id:
            st.session_state["empresa_id"] = ""
            st.session_state["empresa_nome"] = ""
        st.sidebar.success("Empresa deletada.")
        st.rerun()


st.markdown("## 🏢 Empresa (carregar / salvar)")

empresas = listar_empresas()
opcoes = ["— Nova empresa —"] + [f"{nome}  ({eid})" for eid, nome in empresas]

sel = st.selectbox("Selecione uma empresa", opcoes, index=0)

colA, colB, colC = st.columns([2, 1, 1])

with colA:
    nome_novo = ""
    if sel == "— Nova empresa —":
        nome_novo = st.text_input("Nome da nova empresa", placeholder="Ex.: ACME S.A.")
with colB:
    btn_carregar = st.button("📥 Carregar", use_container_width=True)
with colC:
    btn_salvar = st.button("💾 Salvar", use_container_width=True)

# Resolve empresa_id atual
if sel == "— Nova empresa —":
    empresa_nome = nome_novo.strip()
    empresa_id = slugify(empresa_nome) if empresa_nome else ""
else:
    # extrai id entre parênteses no final
    empresa_id = sel.split("(")[-1].replace(")", "").strip()
    empresa_nome = dict(empresas).get(empresa_id, empresa_id)

# Guarda seleção no session_state
st.session_state["empresa_id"] = empresa_id
st.session_state["empresa_nome"] = empresa_nome

# CARREGAR
if btn_carregar:
    if not empresa_id:
        st.warning("Informe o nome da empresa para carregar/criar.")
    else:
        data = carregar_empresa(empresa_id)
        if not data:
            st.info("Empresa ainda não tem dados salvos. Você pode preencher e salvar.")
        else:
            # restaura dataframes nos estados do app
            st.session_state["dre_raw"] = records_to_df(data.get("dre_raw"))
            st.session_state["dre_override"] = records_to_df(data.get("dre_override")).set_index("Conta") if data.get("dre_override") else st.session_state.get("dre_override")
            st.session_state["bp_raw"] = records_to_df(data.get("bp_raw"))
            st.session_state["bp_override"] = records_to_df(data.get("bp_override")).set_index("Conta") if data.get("bp_override") else st.session_state.get("bp_override")
            st.success(f"Dados carregados: {empresa_nome} ({empresa_id})")

# SALVAR
if btn_salvar:
    if not empresa_id:
        st.warning("Informe o nome da empresa antes de salvar.")
    else:
        payload = {
            "empresa_id": empresa_id,
            "empresa_nome": empresa_nome,
            "dre_raw": df_to_records(st.session_state.get("dre_raw")),
            "dre_override": df_to_records(st.session_state.get("dre_override").reset_index()) if isinstance(st.session_state.get("dre_override"), pd.DataFrame) else [],
            "bp_raw": df_to_records(st.session_state.get("bp_raw")),
            "bp_override": df_to_records(st.session_state.get("bp_override").reset_index()) if isinstance(st.session_state.get("bp_override"), pd.DataFrame) else [],
        }
        salvar_empresa(empresa_id, payload)
        st.success(f"Dados salvos: {empresa_nome} ({empresa_id})")


# =========================================================
# CONSTANTES
# =========================================================
anos = [f"Ano {i}" for i in range(1, 7)]

# =========================================================
# FUNÇÕES AUXILIARES (estilo + formatação segura)
# =========================================================

def slugify(nome: str) -> str:
    """Gera um ID seguro a partir do nome da empresa."""
    s = (nome or "").strip().lower()
    s = re.sub(r"[^a-z0-9]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s or "empresa"

def empresa_path(empresa_id: str) -> Path:
    return DATA_DIR / f"{empresa_id}.json"

def df_to_records(df: pd.DataFrame) -> list:
    """Serializa DF para lista de dicts (JSON-friendly)."""
    if df is None:
        return []
    return df.to_dict(orient="records")

def records_to_df(records: list) -> pd.DataFrame:
    """Desserializa lista de dicts para DF."""
    if not records:
        return pd.DataFrame()
    return pd.DataFrame.from_records(records)

def salvar_empresa(empresa_id: str, payload: dict) -> None:
    path = empresa_path(empresa_id)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

def carregar_empresa(empresa_id: str) -> dict | None:
    path = empresa_path(empresa_id)
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)

def listar_empresas() -> list[tuple[str, str]]:
    """
    Retorna lista de (empresa_id, display_name) lendo os arquivos.
    display_name é o que foi salvo no JSON.
    """
    out = []
    for p in sorted(DATA_DIR.glob("*.json")):
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
            out.append((p.stem, data.get("empresa_nome", p.stem)))
        except Exception:
            out.append((p.stem, p.stem))
    return out

def parse_num_br(x):
    """
    Converte entradas típicas pt-BR em float:
    - "1.850.000" -> 1850000
    - "1.850.000,50" -> 1850000.50
    - "1850000" -> 1850000
    - "" / None -> 0.0
    Mantém números já numéricos.
    """
    if x is None:
        return 0.0
    if isinstance(x, (int, float, np.integer, np.floating)):
        return float(x)

    s = str(x).strip()

    if s == "":
        return 0.0

    # remove "R$" e espaços
    s = s.replace("R$", "").replace(" ", "")

    # padrão brasileiro: '.' milhar e ',' decimal
    # remove milhares e troca decimal
    s = s.replace(".", "").replace(",", ".")

    try:
        return float(s)
    except Exception:
        return 0.0


def garantir_numerico_df(df, cols):
    """
    Garante que as colunas numéricas do DF estejam realmente numéricas
    (mesmo que o usuário tenha digitado com ponto/virgula).
    """
    out = df.copy()
    for c in cols:
        out[c] = out[c].apply(parse_num_br).astype(float)
    return out


def formatar_apenas_valores(styler_or_df):
    """
    Recebe um DataFrame ou um Styler e aplica formatação monetária
    somente nas colunas numéricas (evita erro ao tentar formatar a coluna 'Conta').
    """
    df = styler_or_df.data if hasattr(styler_or_df, "data") else styler_or_df
    colunas_numericas = df.select_dtypes(include="number").columns
    formatos = {col: "R$ {:,.0f}" for col in colunas_numericas}

    if hasattr(styler_or_df, "format"):
        return styler_or_df.format(formatos)
    return df.style.format(formatos)

def altura_dataframe(
    df,
    altura_linha=34,
    altura_header=42,
    padding=30,
    max_altura=1400
):
    """
    Calcula altura suficiente para evitar scroll interno no st.dataframe.
    """
    n = len(df)
    altura = altura_header + n * altura_linha + padding
    return min(altura, max_altura)


def _to_num(df, cols):
    out = df.copy()
    for c in cols:
        out[c] = pd.to_numeric(out[c], errors="coerce").fillna(0.0)
    return out


def consolidar_dre_com_override(dre_df: pd.DataFrame, override_df: pd.DataFrame) -> pd.DataFrame:
    """
    Consolida contas principais do DRE por ano e aplica override quando preenchido.

    Regras:
    - Sempre NEGATIVOS (mesmo se usuário digitar +): CMV, Despesas, D&A, Imposto
    - Respeitam sinal digitado (+/-): Outras operacionais, Resultado Financeiro, Outros Não Operacionais
    - Compatível com símbolos na coluna Conta: "(+)", "(-)", "(=)", "(+/-)"
    """

    df = dre_df.copy()

    # Garantir numérico
    for a in anos:
        df[a] = pd.to_numeric(df[a], errors="coerce").fillna(0.0)

    df_idx = df.set_index("Conta")

    # Utilitário: máscara por nome lógico (conta_limpa)
    def mask_conta(nome_logico: str):
        return df_idx.index.to_series().map(conta_limpa).eq(nome_logico).values

    # Getter por nome lógico
    def get(nome_logico: str) -> pd.Series:
        m = mask_conta(nome_logico)
        if m.any():
            return df_idx.loc[m, anos].astype(float).iloc[0]
        return pd.Series({a: 0.0 for a in anos})

    # Setter por nome lógico
    def set_row(nome_logico: str, serie: pd.Series):
        m = mask_conta(nome_logico)
        if m.any():
            for a in anos:
                df_idx.loc[m, a] = float(serie[a])

    # 1) Forçar NEGATIVO nas contas sempre-negativas (AGORA funciona com símbolos)
    sempre_negativas = [
        "CMV, CPV ou CSP",
        "Despesas de Vendas",
        "Despesas gerais e administrativas",
        "Depreciação & Amortização",
        "Imposto de Renda",
    ]
    for nome in sempre_negativas:
        m = mask_conta(nome)
        if m.any():
            for a in anos:
                df_idx.loc[m, a] = -abs(float(df_idx.loc[m, a].iloc[0]))

    # 2) Cálculos (respeitando regra de sinais)
    receita = get("Receita Líquida")
    cmv = get("CMV, CPV ou CSP")  # já negativo
    lucro_bruto = receita + cmv

    desp_vendas = get("Despesas de Vendas")  # negativo
    desp_ga = get("Despesas gerais e administrativas")  # negativo
    outras_oper = get("Outras despesas/receitas operacionais")  # LIVRE
    da = get("Depreciação & Amortização")  # negativo

    # IMPORTANTE: EBIT é antes de juros e impostos; D&A é despesa operacional (entra no EBIT).
    ebit = lucro_bruto + desp_vendas + desp_ga + outras_oper + da

    fin = get("Resultado Financeiro")  # LIVRE
    outros_nonop = get("Outros Resultados Não Operacionais")  # LIVRE
    lair = ebit + fin + outros_nonop

    imposto = get("Imposto de Renda")  # negativo
    lucro_liq = lair + imposto

    # EBITDA = EBIT + D&A (add-back). Como DA é negativo, subtrair DA soma.
    ebitda = ebit - da

    # 3) Escrever totais automáticos
    set_row("Lucro Bruto", lucro_bruto)
    set_row("Lucro Operacional - EBIT", ebit)
    set_row("Lucro Antes do IR", lair)
    set_row("Lucro Líquido", lucro_liq)
    set_row("EBITDA", ebitda)

    # 4) Override (também por conta_limpa)
    if override_df is not None and not override_df.empty:
        # override_df tem index "Conta" (sem símbolos, como você cadastrou)
        for total in ["Lucro Bruto", "Lucro Operacional - EBIT", "Lucro Antes do IR", "Lucro Líquido", "EBITDA"]:
            if total in override_df.index:
                m = mask_conta(total)
                if m.any():
                    for a in anos:
                        ovr = override_df.loc[total, a]
                        if pd.notna(ovr):
                            df_idx.loc[m, a] = float(ovr)

    return df_idx.reset_index()



def criar_override_df(contas_consolidadoras: list, anos: list) -> pd.DataFrame:
    ov = pd.DataFrame(index=contas_consolidadoras, columns=anos, data=np.nan)
    ov.index.name = "Conta"
    return ov

def consolidar_bp_com_override(bp_df: pd.DataFrame, override_df: pd.DataFrame) -> pd.DataFrame:
    """
    Consolida contas principais do BP por soma de linhas-filhas e aplica override.
    Linhas separadoras (Conta vazia) não entram nos cálculos.
    """
    df = bp_df.copy()

    for a in anos:
        df[a] = pd.to_numeric(df[a], errors="coerce").fillna(0.0)

    df_idx = df.set_index("Conta")

    # Remove linhas vazias/separadoras APENAS dos cálculos
    df_idx_calc = df_idx[df_idx.index.astype(str).str.strip() != ""]

    mapa = {
        "Ativo Circulante": [
            "Caixa e Similares", "Contas a Receber", "Estoques", "Adiantamentos", "Outros ativos circulantes"
        ],
        "Ativo Não Circulante": [
            "Investimentos em Outras Cias", "Imobilizado", "Intangível", "Propriedades para Investimentos"
        ],
        "Passivo Circulante": [
            "Empréstimos e Financiamentos (CP)", "Fornecedores", "Salários",
            "Impostos e Encargos Sociais", "Outros Passivos Circulantes"
        ],
        "Passivo Não Circulante": [
            "Empréstimos e Financiamentos (LP)", "Impostos (LP)", "Outras Contas a Pagar"
        ],
        "Patrimônio Líquido": [
            "Capital Social", "Reserva de Lucros", "Resultados Acumulados"
        ],
    }

    for total, comps in mapa.items():
        if total not in df_idx_calc.index:
            continue

        comps_exist = [c for c in comps if c in df_idx_calc.index]
        if not comps_exist:
            continue

        for a in anos:
            auto = float(df_idx_calc.loc[comps_exist, a].sum())

            if total in override_df.index:
                ovr = override_df.loc[total, a]
                df_idx.loc[total, a] = float(ovr) if pd.notna(ovr) else auto
            else:
                df_idx.loc[total, a] = auto

    return df_idx.reset_index()


def destacar_dre(df):
    contas_bold = {
        "Receita Líquida",
        "Lucro Bruto",
        "Lucro Operacional - EBIT",
        "Lucro Antes do IR",
        "Lucro Líquido",
        "EBITDA",
    }

    def style_row(row):
        nome = conta_limpa(row["Conta"])
        if nome in contas_bold:
            return ["font-weight: bold"] * len(row)
        return [""] * len(row)

    return df.style.apply(style_row, axis=1)


def conta_limpa(nome):
    if nome is None:
        return ""
    nome = str(nome)
    for p in ["(+/-)", "(+)", "(-)", "(=)"]:
        nome = nome.replace(p, "")
    return nome.strip()

def get_val(df: pd.DataFrame, conta: str, ano: str) -> float:
    """
    Busca valor (float) por nome lógico da conta (independente de símbolos).
    Retorna 0.0 se não encontrar.
    """
    if df is None or df.empty:
        return 0.0
    s_conta = df["Conta"].astype(str).map(conta_limpa)
    mask = (s_conta == conta)
    if not mask.any():
        return 0.0
    v = df.loc[mask, ano].iloc[0]
    try:
        return float(v)
    except Exception:
        return 0.0

def delta(df: pd.DataFrame, conta: str, ano_atual: str, ano_anterior: str) -> float:
    return get_val(df, conta, ano_atual) - get_val(df, conta, ano_anterior)


# =========================================================
# TABS
# =========================================================
tab1, tab2, tab3, tab4 = st.tabs(["📥 Banco de Dados", "📈 Análises Financeiras", "🧩 Matriz do Caixa", "🧪 Simulações"])

# =========================================================
# TAB 1 — BANCO DE DADOS
# =========================================================
with tab1:
    st.subheader("📥 Banco de Dados — Instruções Importantes")

    st.info(
        "➡️ **Como preencher os períodos**\n\n"
        "- **Ano 1**: período mais antigo\n"
        "- **Ano 6**: período mais recente\n\n"
        "Preencha sempre **da esquerda para a direita**.\n"
        "As análises utilizarão automaticamente o **último período preenchido**."
    )

    st.divider()

    # -----------------------------------------------------
    # SUBABAS
    # -----------------------------------------------------
    subtab_edit, subtab_view, subtab_cashflow = st.tabs(["✍️ Preenchimento", "👁️ Visualização", "💧 Fluxo de Caixa"])

    # =====================================================
    # SUBABA — PREENCHIMENTO
    # =====================================================
    with subtab_edit:

        # =========================
        # DRE
        # =========================
        st.subheader("Demonstração do Resultado (DRE)")

        dre_contas = [
            "(+) Receita Líquida",
            "(-) CMV, CPV ou CSP",
            "(=) Lucro Bruto",
            "(-) Despesas de Vendas",
            "(-) Despesas gerais e administrativas",
            "(+/-) Outras despesas/receitas operacionais",
            "(-) Depreciação & Amortização",
            "(=) Lucro Operacional - EBIT",
            "(+/-) Resultado Financeiro",
            "(+/-) Outros Resultados Não Operacionais",
            "(=) Lucro Antes do IR",
            "(-) Imposto de Renda",
            "(=) Lucro Líquido",
            "EBITDA",
        ]

        # Inicialização
        if "dre_raw" not in st.session_state:
            df = pd.DataFrame({"Conta": dre_contas})
            for a in anos:
                df[a] = ""
            st.session_state["dre_raw"] = df
        if not isinstance(st.session_state["dre_raw"], pd.DataFrame):
            try:
                st.session_state["dre_raw"] = pd.DataFrame(st.session_state["dre_raw"])
            except Exception:
                df = pd.DataFrame({"Conta": dre_contas})
                for a in anos:
                    df[a] = ""
                st.session_state["dre_raw"] = df
        if st.session_state["dre_raw"].empty:
            df = pd.DataFrame({"Conta": dre_contas})
            for a in anos:
                df[a] = ""
            st.session_state["dre_raw"] = df

        # Editor (NUNCA converter antes)
        dre_raw = st.data_editor(
            st.session_state["dre_raw"],
            disabled=["Conta"],
            num_rows="fixed",
            use_container_width=True,
            height=altura_dataframe(st.session_state["dre_raw"]),
            key="dre_editor"
        )
        st.session_state["dre_raw"] = dre_raw.copy()

        # Override
        contas_override_dre = [
            "Lucro Bruto",
            "Lucro Operacional - EBIT",
            "Lucro Antes do IR",
            "Lucro Líquido",
            "EBITDA",
        ]

        if "dre_override" not in st.session_state:
            st.session_state["dre_override"] = criar_override_df(contas_override_dre, anos)

        dre_override = st.data_editor(
            st.session_state["dre_override"].reset_index(),
            disabled=["Conta"],
            num_rows="fixed",
            use_container_width=True,
            height=altura_dataframe(st.session_state["dre_override"].reset_index()),
            key="dre_override_editor"
        ).set_index("Conta")

        st.session_state["dre_override"] = dre_override.copy()

        # Conversão SOMENTE para cálculo
        dre_num = garantir_numerico_df(dre_raw, anos)
        dre_override_num = garantir_numerico_df(dre_override.reset_index(), anos).set_index("Conta")

        # Consolidação
        st.session_state["dre_df"] = consolidar_dre_com_override(dre_num, dre_override_num)

        st.divider()

        # =========================
        # BALANÇO
        # =========================
        st.subheader("Balanço Patrimonial")

        balanco_contas = [
            "Caixa e Similares",
            "Contas a Receber",
            "Estoques",
            "Adiantamentos",
            "Outros ativos circulantes",
            "Ativo Circulante",
            " ",
            "Investimentos em Outras Cias",
            "Imobilizado",
            "Intangível",
            "Propriedades para Investimentos",
            "Ativo Não Circulante",
            " ",
            "Empréstimos e Financiamentos (CP)",
            "Fornecedores",
            "Salários",
            "Impostos e Encargos Sociais",
            "Outros Passivos Circulantes",
            "Passivo Circulante",
            " ",
            "Empréstimos e Financiamentos (LP)",
            "Impostos (LP)",
            "Outras Contas a Pagar",
            "Passivo Não Circulante",
            " ",
            "Capital Social",
            "Reserva de Lucros",
            "Resultados Acumulados",
            "Patrimônio Líquido",
        ]

        if "bp_raw" not in st.session_state:
            df = pd.DataFrame({"Conta": balanco_contas})
            for a in anos:
                df[a] = ""
            st.session_state["bp_raw"] = df
        if not isinstance(st.session_state["bp_raw"], pd.DataFrame):
            try:
                st.session_state["bp_raw"] = pd.DataFrame(st.session_state["bp_raw"])
            except Exception:
                df = pd.DataFrame({"Conta": balanco_contas})
                for a in anos:
                    df[a] = ""
                st.session_state["bp_raw"] = df
        if st.session_state["bp_raw"].empty:
            df = pd.DataFrame({"Conta": balanco_contas})
            for a in anos:
                df[a] = ""
            st.session_state["bp_raw"] = df

        bp_raw = st.data_editor(
            st.session_state["bp_raw"],
            disabled=["Conta"],
            num_rows="fixed",
            use_container_width=True,
            height=altura_dataframe(st.session_state["bp_raw"]),
            key="bp_editor"
        )
        st.session_state["bp_raw"] = bp_raw.copy()

        contas_override_bp = [
            "Ativo Circulante",
            "Ativo Não Circulante",
            "Passivo Circulante",
            "Passivo Não Circulante",
            "Patrimônio Líquido",
        ]

        if "bp_override" not in st.session_state:
            st.session_state["bp_override"] = criar_override_df(contas_override_bp, anos)

        bp_override = st.data_editor(
            st.session_state["bp_override"].reset_index(),
            disabled=["Conta"],
            num_rows="fixed",
            use_container_width=True,
            height=altura_dataframe(st.session_state["bp_override"].reset_index()),
            key="bp_override_editor"
        ).set_index("Conta")

        st.session_state["bp_override"] = bp_override.copy()

        bp_num = garantir_numerico_df(bp_raw, anos)
        bp_override_num = garantir_numerico_df(bp_override.reset_index(), anos).set_index("Conta")

        st.session_state["balanco_df"] = consolidar_bp_com_override(bp_num, bp_override_num)



    # =====================================================
    # SUBABA — VISUALIZAÇÃO
    # =====================================================
    with subtab_view:
        st.subheader("👁️ Visualização Estruturada")

        # --------- DRE ----------
        contas_consolidadoras_dre_view = [
            "Receita Líquida",
            "Lucro Bruto",
            "Lucro Operacional - EBIT",
            "Lucro Antes do IR",
            "Lucro Líquido",
            "EBITDA",
        ]

        st.markdown("### DRE — Estrutura (com consolidação automática)")

        df_dre_view = st.session_state.get("dre_df", pd.DataFrame(columns=["Conta"] + anos))
        st.dataframe(
            formatar_apenas_valores(destacar_dre(df_dre_view)),
            use_container_width=True,
            height=altura_dataframe(df_dre_view)
        )

        st.divider()

        # --------- BALANÇO ----------
        contas_consolidadoras_bp_view = [
            "Ativo Circulante",
            "Ativo Não Circulante",
            "Passivo Circulante",
            "Passivo Não Circulante",
            "Patrimônio Líquido",
        ]

        def destacar_bp(df):
            def style_row(row):
                conta = str(row["Conta"]) if row["Conta"] is not None else ""
                if conta.strip() == "":
                    return ["background-color: white"] * len(row)
                if row["Conta"] in contas_consolidadoras_bp_view:
                    return ["font-weight: bold"] * len(row)
                return [""] * len(row)
            return df.style.apply(style_row, axis=1)


        st.markdown("### Balanço Patrimonial — Estrutura (com consolidação automática)")

        df_bp_view = st.session_state.get("balanco_df", pd.DataFrame(columns=["Conta"] + anos))
        st.dataframe(
            formatar_apenas_valores(destacar_bp(df_bp_view)),
            use_container_width=True,
            height=altura_dataframe(df_bp_view)
        )


    with subtab_cashflow:
        
        st.markdown("### 💧 Proxy do Fluxo de Caixa (Método Indireto)")

        dre_df = st.session_state.get("dre_df")
        bp_df  = st.session_state.get("balanco_df")

        if dre_df is None or bp_df is None or dre_df.empty or bp_df.empty:
            st.warning("Preencha DRE e Balanço na aba 'Banco de Dados' para gerar o Fluxo de Caixa.")
        else:
            st.caption(
                "Modelo gerencial (indireto) construído a partir de DRE + BP. "
                "CFI e CFF são proxies por variação patrimonial; a conciliação com a variação de Caixa mostra a diferença."
            )

            # -------------------------------------------------
            # Monta FC para 5 períodos: Ano2-Ano1 ... Ano6-Ano5
            # -------------------------------------------------
            linhas = []

            for i in range(2, 7):
                a_atual = f"Ano {i}"
                a_ant   = f"Ano {i-1}"
                periodo = f"{a_ant} → {a_atual}"

                # ---------
                # DRE (ano atual) para LL e D&A
                # ---------
                ll = get_val(dre_df, "Lucro Líquido", a_atual)
                da = get_val(dre_df, "Depreciação & Amortização", a_atual)

                # Observação: no seu modelo, D&A é forçado negativo.
                # No CFO indireto, somamos de volta o efeito não-caixa:
                da_addback = abs(da)

                # ---------
                # BP deltas (atual - anterior)
                # ---------
                d_caixa = delta(bp_df, "Caixa e Similares", a_atual, a_ant)

                # Ativo Circulante operacional (sem caixa) — aumento consome caixa
                d_cr   = delta(bp_df, "Contas a Receber", a_atual, a_ant)
                d_est  = delta(bp_df, "Estoques", a_atual, a_ant)
                d_adi  = delta(bp_df, "Adiantamentos", a_atual, a_ant)
                d_out_ac = delta(bp_df, "Outros ativos circulantes", a_atual, a_ant)

                # Passivo circulante operacional — aumento gera caixa
                d_forn = delta(bp_df, "Fornecedores", a_atual, a_ant)
                d_sal  = delta(bp_df, "Salários", a_atual, a_ant)
                d_imp  = delta(bp_df, "Impostos e Encargos Sociais", a_atual, a_ant)
                d_out_pc = delta(bp_df, "Outros Passivos Circulantes", a_atual, a_ant)

                # Dívida (financiamento) — aumentos geram caixa
                d_div_cp = delta(bp_df, "Empréstimos e Financiamentos (CP)", a_atual, a_ant)
                d_div_lp = delta(bp_df, "Empréstimos e Financiamentos (LP)", a_atual, a_ant)
                d_divida = d_div_cp + d_div_lp

                # Ativo não circulante (proxy de investimento)
                d_invest = delta(bp_df, "Investimentos em Outras Cias", a_atual, a_ant)
                d_imob   = delta(bp_df, "Imobilizado", a_atual, a_ant)
                d_intang = delta(bp_df, "Intangível", a_atual, a_ant)
                d_prop   = delta(bp_df, "Propriedades para Investimentos", a_atual, a_ant)
                d_anc_proxy = d_invest + d_imob + d_intang + d_prop

                # Patrimônio líquido (proxy de captação/retorno ao acionista)
                d_cap  = delta(bp_df, "Capital Social", a_atual, a_ant)
                d_res  = delta(bp_df, "Reserva de Lucros", a_atual, a_ant)
                d_ret  = delta(bp_df, "Resultados Acumulados", a_atual, a_ant)
                d_pl_proxy = d_cap + d_res + d_ret

                # -------------------------------------------------
                # CFO (Indireto) — básico e robusto
                # -------------------------------------------------
                delta_wc = (d_cr + d_est + d_adi + d_out_ac) - (d_forn + d_sal + d_imp + d_out_pc)
                # Aumento de WC consome caixa (subtrai)
                cfo = ll + da_addback - delta_wc

                # -------------------------------------------------
                # CFI (Investimentos) — proxy pela variação do ANC
                # Se ANC aumenta => consumo de caixa => negativo
                # -------------------------------------------------
                cfi = -d_anc_proxy

                # -------------------------------------------------
                # CFF (Financiamentos) — proxy por dívida + PL
                # Aumento dívida/PL => entrada de caixa => positivo
                # -------------------------------------------------
                cff = d_divida + d_pl_proxy

                # Variação de caixa "calculada"
                d_caixa_calc = cfo + cfi + cff

                linhas.append({
                    "Período": periodo,
                    "Lucro Líquido (DRE)": ll,
                    "D&A": da_addback,
                    "Δ Ativo Circulante": delta_wc,
                    "Fluxo de Caixa Operacional": cfo,
                    "Fluxo de Caixa de Invesimento (Δ ANC)": cfi,
                    "Fluxo de Caixa de Financiamento": cff,
                    "Δ Caixa (calculado)": d_caixa_calc,
                    "Δ Caixa (BP)": d_caixa,
                    "Diferença (calc - BP)": d_caixa_calc - d_caixa
                })

            df_fc = pd.DataFrame(linhas)

            df_fc_t = (
                df_fc
                .set_index("Período")
                .T
                .reset_index()
                .rename(columns={"index": "Conta"})
            )

            st.dataframe(
                df_fc_t.style.format({
                    col: "R$ {:,.0f}" for col in df_fc_t.columns if col != "Conta"
                }),
                use_container_width=True,
                height=min(900, 40 + 35 * (len(df_fc_t) + 2))
            )


            st.divider()
            st.markdown("#### Leitura rápida")
            st.write(
                "- **CFO**: lucro líquido somado por D&A e variação do capital de giro (Δ AC).\n"
                "- **CFI**: proxy por variação do Ativo Não Circulante.\n"
                "- **CFF**: proxy por variação de dívida e PL.\n"
                "- **Diferença**: mostra o quanto o modelo gerencial diverge da variação de caixa do BP."
            )



# =========================================================
# TAB 2 — ANÁLISES
# =========================================================
with tab2:
    st.subheader("📈 Análises Financeiras")

    dre_df = st.session_state.get("dre_df")
    bp_df  = st.session_state.get("balanco_df")

    if dre_df is None or bp_df is None or dre_df.empty or bp_df.empty:
        st.warning("Preencha DRE e Balanço na aba 'Banco de Dados' para habilitar as análises.")
    else:
        # -------------------------------------------------
        # Helpers (compatível com símbolos na coluna Conta)
        # -------------------------------------------------
        def _conta_col(df):
            return df["Conta"].astype(str).map(conta_limpa)

        def get_serie(df, conta):
            s = _conta_col(df)
            mask = (s == conta)
            if not mask.any():
                return pd.Series({a: 0.0 for a in anos})
            out = df.loc[mask, anos].iloc[0]
            return pd.to_numeric(out, errors="coerce").fillna(0.0)

        def safe_div(n, d):
            n = np.asarray(n, dtype=float)
            d = np.asarray(d, dtype=float)
            return np.where(d == 0, np.nan, n / d)

        def anos_preenchidos(df):
            cols = []
            for a in anos:
                col = pd.to_numeric(df[a], errors="coerce").fillna(0.0)
                if float(col.abs().sum()) != 0.0:
                    cols.append(a)
            return cols

        anos_ok = sorted(set(anos_preenchidos(dre_df)) | set(anos_preenchidos(bp_df)),
                         key=lambda x: int(x.split()[-1]))
        if len(anos_ok) == 0:
            anos_ok = anos[:]

        def cagr(v0, v1, n_int):
            if n_int <= 0 or v0 is None or v1 is None:
                return None
            v0 = float(v0); v1 = float(v1)
            if v0 <= 0 or v1 <= 0:
                return None
            return (v1 / v0) ** (1 / n_int) - 1

        def first_last_and_nint(serie: pd.Series):
            vals = [float(serie[a]) for a in anos_ok]
            idx = [i for i, v in enumerate(vals) if v != 0.0]
            if len(idx) < 2:
                return 0.0, 0.0, 0
            i0, i1 = idx[0], idx[-1]
            return vals[i0], vals[i1], (i1 - i0)

        # -------------------------------------------------
        # Subabas
        # -------------------------------------------------
        sub_indic, sub_avah, sub_ciclos, sub_tes, sub_wacc = st.tabs([
            "📈 Indicadores & CAGR",
            "📊 Vertical & Horizontal",
            "⏱️ Ciclos e Alavancagem",
            "🏦 Tesouraria",
            "💰 WACC & Valor"
        ])

        # =================================================
        # SUBABA 0 — Indicadores + Gráfico + CAGR (LIMPA)
        # =================================================
        with sub_indic:
            st.markdown("## 📈 Evolução de Indicadores")
            st.caption("Selecione indicadores para ver a evolução por período.")

            # -----------------------------
            # Séries base (DRE / BP)
            # -----------------------------
            receita  = get_serie(dre_df, "Receita Líquida")
            cmv      = get_serie(dre_df, "CMV, CPV ou CSP").abs()           # usa módulo
            ebit     = get_serie(dre_df, "Lucro Operacional - EBIT")
            ebitda   = get_serie(dre_df, "EBITDA")
            lucroliq = get_serie(dre_df, "Lucro Líquido")
            imposto  = get_serie(dre_df, "Imposto de Renda").abs()          # módulo
            da       = get_serie(dre_df, "Depreciação & Amortização").abs() # módulo

            caixa = get_serie(bp_df, "Caixa e Similares")
            cr    = get_serie(bp_df, "Contas a Receber")
            est   = get_serie(bp_df, "Estoques")
            adi   = get_serie(bp_df, "Adiantamentos")

            ac    = get_serie(bp_df, "Ativo Circulante")
            anc   = get_serie(bp_df, "Ativo Não Circulante")

            forn  = get_serie(bp_df, "Fornecedores")
            sal   = get_serie(bp_df, "Salários")
            impcp = get_serie(bp_df, "Impostos e Encargos Sociais")

            pc    = get_serie(bp_df, "Passivo Circulante")
            pnc   = get_serie(bp_df, "Passivo Não Circulante")
            pl    = get_serie(bp_df, "Patrimônio Líquido")

            # Dívida (proxy)
            div_cp = get_serie(bp_df, "Empréstimos e Financiamentos (CP)")
            div_lp = get_serie(bp_df, "Empréstimos e Financiamentos (LP)")
            div_total = div_cp + div_lp
            div_liq   = div_total - caixa

            # -----------------------------
            # Tesouraria (Fleuriet)
            # CPL = (PNC + PL) - ANC
            # IOG = ACC - PCC
            # ACC = CR + Estoques + Adiantamentos
            # PCC = Forn + Sal + Impostos Encargos
            # -----------------------------
            acc = cr + est + adi
            pcc = forn + sal + impcp
            iog = acc - pcc
            cpl = (pnc + pl) - anc
            saldo_tes = iog - cpl

            # -----------------------------
            # FCO (proxy) — método indireto simples
            # FCO = Lucro Líq + D&A - ΔNWC   (NWC = ACC - PCC)
            # -----------------------------
            nwc = acc - pcc
            fco = pd.Series({a: np.nan for a in anos_ok})
            for j, a in enumerate(anos_ok):
                if j == 0:
                    fco[a] = np.nan
                else:
                    a_prev = anos_ok[j - 1]
                    delta_nwc = float(nwc[a]) - float(nwc[a_prev])
                    fco[a] = float(lucroliq[a] + da[a] - delta_nwc)

            # -----------------------------
            # ROIC (proxy coerente)
            # NOPAT = EBIT * (1 - alíquota)
            # Capital Investido (proxy) = IOG + ANC
            # -----------------------------
            ir_eff = st.number_input("Alíquota efetiva para ROIC (IR/CSLL) %", value=34.0, step=1.0) / 100.0

            def _safe_div_scalar(n, d):
                try:
                    n = float(n); d = float(d)
                    return np.nan if d == 0 else (n / d)
                except Exception:
                    return np.nan

            roic = pd.Series({a: np.nan for a in anos_ok})
            for a in anos_ok:
                nopat = float(ebit[a]) * (1 - ir_eff)
                cap_inv = float(iog[a]) + float(anc[a])
                roic[a] = _safe_div_scalar(nopat, cap_inv) * 100.0

            # ----------------------------
            # Indicadores (BÁSICOS + AVANÇADOS)
            # ----------------------------
            indic = {}

            # Margens
            indic["Margem Bruta (%)"]   = safe_div((receita - cmv), receita) * 100.0
            indic["Margem EBIT (%)"]    = safe_div(ebit, receita) * 100.0
            indic["Margem EBITDA (%)"]  = safe_div(ebitda, receita) * 100.0
            indic["Margem Líquida (%)"] = safe_div(lucroliq, receita) * 100.0

            # Endividamento
            indic["Dívida Total / PL (x)"]       = safe_div(div_total, pl)
            indic["Dívida Líquida / PL (x)"]     = safe_div(div_liq, pl)
            indic["Dívida Líquida / EBITDA (x)"] = safe_div(div_liq, ebitda)
            indic["Participação de Capitais de Terceiros (%)"] = (safe_div(pc + pnc, pc + pnc + pl) * 100.0
)

            # Liquidez
            indic["Liquidez Corrente (AC/PC)"]      = safe_div(ac, pc)
            indic["Liquidez Seca ((AC-Est)/PC)"]    = safe_div((ac - est), pc)
            indic["Caixa / Dívida de CP (%)"]       = safe_div(caixa, div_cp) * 100.0

            # Eficiência (múltiplos)
            indic["Contas Receber / Receita (%)"]        = safe_div(cr, receita) *100.0
            indic["Estoques / CMV (%)"]      = safe_div(est, cmv) *100.0
            indic["Fornecedores / CMV (%)"]  = safe_div(forn, cmv) *100.0

            # Avançados pedidos
            indic["FCO / Dívida Líquida (%)"]    = safe_div(fco, div_liq) *100.0
            indic["FCO / Receita Líquida (%)"]   = safe_div(fco, receita) *100.0
            indic["ROIC (%)"]                   = roic
            indic["ROE (%)"]                   = safe_div(lucroliq, pl) *100.0

            # ----------------------------
            # Tabela (linhas = indicadores, colunas = anos_ok)
            # ----------------------------
            df_ind = pd.DataFrame({k: v for k, v in indic.items()}).T
            df_ind = df_ind[anos_ok]
            df_ind.index.name = "Indicador"

            st.markdown("#### Tabela de Indicadores")

            # ----------------------------
            # Formatação compatível (sem axis no Styler)
            # ----------------------------
            df_ind_fmt = df_ind.copy()

            def _fmt_x(v):
                if v is None or (isinstance(v, float) and np.isnan(v)):
                    return ""
                return f"{float(v):.2f}x"

            def _fmt_pct(v):
                if v is None or (isinstance(v, float) and np.isnan(v)):
                    return ""
                return f"{float(v):.2f}%"

            def _fmt_rs(v):
                if v is None or (isinstance(v, float) and np.isnan(v)):
                    return ""
                return f"R$ {float(v):,.0f}"

            def _fmt_num(v):
                if v is None or (isinstance(v, float) and np.isnan(v)):
                    return ""
                return f"{float(v):.2f}"

            # Aplica formatação por linha (indicador)
            for idx in df_ind_fmt.index:
                nome = str(idx)
                if "(x)" in nome:
                    df_ind_fmt.loc[idx, :] = df_ind_fmt.loc[idx, :].apply(_fmt_x)
                elif "(%)" in nome or "%" in nome:
                    df_ind_fmt.loc[idx, :] = df_ind_fmt.loc[idx, :].apply(_fmt_pct)
                elif "(R$)" in nome or nome.endswith("(R$)"):
                    df_ind_fmt.loc[idx, :] = df_ind_fmt.loc[idx, :].apply(_fmt_rs)
                else:
                    df_ind_fmt.loc[idx, :] = df_ind_fmt.loc[idx, :].apply(_fmt_num)

            st.dataframe(
                df_ind_fmt,
                use_container_width=True,
                height=min(780, 40 + 32 * (len(df_ind_fmt) + 2))
            )


            st.divider()

            # ----------------------------
            # Gráfico selecionável (mantém multi-seleção)
            # ----------------------------
            st.markdown("#### Gráfico — evolução do indicador")

            opcoes = list(df_ind.index)

            indicadores_sel = st.multiselect(
                "Selecione os indicadores para plotar",
                options=opcoes,
                default=opcoes[:3] if len(opcoes) >= 3 else opcoes
            )

            fig = go.Figure()
            for ind in indicadores_sel:
                y = [float(df_ind.loc[ind, a]) if pd.notna(df_ind.loc[ind, a]) else np.nan for a in anos_ok]
                fig.add_trace(go.Scatter(x=anos_ok, y=y, mode="lines+markers", name=str(ind)))

            fig.update_layout(
                height=520,
                xaxis_title="Período",
                yaxis_title="Valor",
                legend_title="Indicadores",
                margin=dict(l=10, r=10, t=10, b=10)
            )

            st.plotly_chart(fig, use_container_width=True)

            # ----------------------------
            # Cards de CAGR (pontos importantes)
            # ----------------------------
            st.divider()
            st.markdown("#### CAGR — pontos-chave (do primeiro ao último ano preenchido)")

            cagr_series = {
                "Faturamento (Receita)": receita,
                "EBITDA": ebitda,
                "Lucro Líquido": lucroliq,
                "Caixa": caixa,
                "Dívida Total": div_total,
                "Dívida Líquida": div_liq,
            }

            cards = st.columns(6)
            for i, (nome, s) in enumerate(cagr_series.items()):
                v0, v1, nint = first_last_and_nint(s)
                g = cagr(v0, v1, nint)
                txt = "n/a" if g is None else f"{g*100:,.1f}%"
                cards[i].metric(nome, txt, help="CAGR do primeiro ao último período preenchido (valores > 0).")

        # =================================================
        # SUBABA 1 — Vertical & Horizontal
        # =================================================
        with sub_avah:
            st.markdown("### 📊 Análise Vertical e Horizontal")

            alvo = st.selectbox("Escolha a demonstração", ["DRE", "Balanço Patrimonial"], index=0)

            if alvo == "DRE":
                df_base = dre_df.copy()
                base_nome = "Receita Líquida"
                base = get_serie(df_base, "Receita Líquida")
            else:
                df_base = bp_df.copy()
                base_nome = "Ativo Total (AC + ANC)"
                base = get_serie(df_base, "Ativo Circulante") + get_serie(df_base, "Ativo Não Circulante")

            for a in anos:
                df_base[a] = pd.to_numeric(df_base[a], errors="coerce").fillna(0.0)

            # -------- Vertical (%)
            df_vert = df_base[["Conta"] + anos_ok].copy()
            for a in anos_ok:
                df_vert[a] = safe_div(df_vert[a].values, float(base[a])) * 100.0

            # -------- Horizontal (% var)
            df_hpct = df_base[["Conta"] + anos_ok].copy()
            for j in range(1, len(anos_ok)):
                a_now = anos_ok[j]
                a_prev = anos_ok[j-1]
                abs_var = df_base[a_now] - df_base[a_prev]
                pct_var = safe_div(abs_var.values, df_base[a_prev].values) * 100.0
                df_hpct[a_now] = pct_var

            if len(anos_ok) >= 1:
                df_hpct[anos_ok[0]] = np.nan

            c1, c2 = st.columns(2)
            with c1:
                st.markdown(f"#### Vertical (%) — Base: {base_nome}")
                st.dataframe(
                    df_vert.style.format({a: "{:,.2f}%" for a in anos_ok}),
                    use_container_width=True,
                    height=min(1200, 40 + 32 * (len(df_vert) + 2))
                )

            with c2:
                st.markdown("#### Horizontal (Δ % vs período anterior)")
                st.dataframe(
                    df_hpct.style.format({a: "{:,.2f}%" for a in anos_ok}),
                    use_container_width=True,
                    height=min(1200, 40 + 32 * (len(df_hpct) + 2))
                )

        # =================================================
        # SUBABA 2 — PMR / PME / PMP
        # =================================================
        with sub_ciclos:
            st.markdown("### ⏱️ Ciclo de Caixa — PMR, PME, PMP")

            st.caption(
                "Premissas:\n"
                "- **PMR** = Contas a Receber / Receita Líquida × 365\n"
                "- **PME** = Estoques / CMV × 365\n"
                "- **PMP** = Fornecedores / CMV × 365\n"
                "- **Ciclo Operacional** = PMR + PME\n"
                "- **Ciclo Financeiro** = (PMR + PME) - PMP\n"
                "Obs.: CMV é usado em módulo (se estiver negativo)."
            )

            # -----------------------------
            # Séries base (DRE / BP)
            # -----------------------------
            receita = get_serie(dre_df, "Receita Líquida")
            cmv = get_serie(dre_df, "CMV, CPV ou CSP").abs()

            cr = get_serie(bp_df, "Contas a Receber")
            est = get_serie(bp_df, "Estoques")
            forn = get_serie(bp_df, "Fornecedores")

            def safe_div_scalar(n, d):
                try:
                    n = float(n); d = float(d)
                    if d == 0:
                        return np.nan
                    return n / d
                except Exception:
                    return np.nan

            # -----------------------------
            # Calcula PMR/PME/PMP por ano (anos_ok)
            # -----------------------------
            pmr_s, pme_s, pmp_s, cop_s, cfi_s = {}, {}, {}, {}, {}
            for a in anos_ok:
                pmr = safe_div_scalar(cr[a], receita[a]) * 365.0 if float(receita[a]) != 0 else np.nan
                pme = safe_div_scalar(est[a], cmv[a]) * 365.0 if float(cmv[a]) != 0 else np.nan
                pmp = safe_div_scalar(forn[a], cmv[a]) * 365.0 if float(cmv[a]) != 0 else np.nan

                cop = (pmr + pme) if (pd.notna(pmr) and pd.notna(pme)) else np.nan
                cfi = (cop - pmp) if (pd.notna(cop) and pd.notna(pmp)) else np.nan

                pmr_s[a], pme_s[a], pmp_s[a], cop_s[a], cfi_s[a] = pmr, pme, pmp, cop, cfi

            df_ciclos = pd.DataFrame({
                "PMR (dias)": pd.Series(pmr_s),
                "PME (dias)": pd.Series(pme_s),
                "PMP (dias)": pd.Series(pmp_s),
                "Ciclo Operacional": pd.Series(cop_s),
                "Ciclo Financeiro": pd.Series(cfi_s),
            }).T
            df_ciclos.index.name = "Indicador"
            df_ciclos = df_ciclos[anos_ok]

            st.dataframe(
                df_ciclos.style.format({a: "{:,.0f}" for a in anos_ok}),
                use_container_width=True,
                height=min(520, 40 + 32 * (len(df_ciclos) + 2))
            )

            st.divider()

            # =========================================================
            # 1) RÉGUA (último ano) — substitui cards
            # =========================================================
            st.markdown("#### Régua — último período (dias)")

            ano_ref = st.selectbox("Selecione o período de referência", options=anos_ok, index=len(anos_ok)-1)

            pmr_u = float(pmr_s.get(ano_ref, np.nan))
            pme_u = float(pme_s.get(ano_ref, np.nan))
            pmp_u = float(pmp_s.get(ano_ref, np.nan))
            cop_u = float(cop_s.get(ano_ref, np.nan))
            cfi_u = float(cfi_s.get(ano_ref, np.nan))

            # fallback se algo vier NaN
            pmr_u = 0.0 if not np.isfinite(pmr_u) else pmr_u
            pme_u = 0.0 if not np.isfinite(pme_u) else pme_u
            pmp_u = 0.0 if not np.isfinite(pmp_u) else pmp_u

            # Recalcula por segurança
            cop_u = pmr_u + pme_u
            cfi_u = cop_u - pmp_u

            # ------------------------------------------------------------------
            # NOVO REFERENCIAL:
            # Eixo passa a começar no "pagamento ao fornecedor" (PMP = 0)
            # Isso deixa a régua mais intuitiva como "linha do tempo"
            # ------------------------------------------------------------------

            # Eventos na régua (a partir do pagamento)
            x_pgto_forn = 0.0                     # agora é o marco inicial (PMP)
            x_fim_estoque = pme_u                 # após PME
            x_recebimento = pme_u + pmr_u         # após PME + PMR (fim do ciclo operacional)
            x_fim_ciclo_fin = cfi_u               # fim do ciclo financeiro (COP - PMP)

            fig_regua = go.Figure()

            # Barras: Operacional e Financeiro (ambas iniciam em 0 neste novo eixo)
            fig_regua.add_trace(go.Bar(
                x=[cop_u],
                y=["Ciclo Operacional"],
                orientation="h",
                name="Ciclo Operacional",
                marker=dict(opacity=0.60),
                base=0
            ))
            fig_regua.add_trace(go.Bar(
                x=[cfi_u],
                y=["Ciclo Financeiro"],
                orientation="h",
                name="Ciclo Financeiro",
                marker=dict(opacity=0.35),
                base=0
            ))

            # Marcadores (com textos)
            fig_regua.add_trace(go.Scatter(
                x=[x_pgto_forn, x_fim_estoque, x_recebimento, x_fim_ciclo_fin],
                y=["Evento"] * 4,
                mode="markers+text",
                text=[
                    "Pgto ao Fornecedor (PMP = 0)",
                    "Fim do Estoque (PME)",
                    "Recebimento (PME+PMR)",
                    "Fim Ciclo Financeiro (COP-PMP)"
                ],
                textposition="top right",
                marker=dict(size=8),
                showlegend=False,
                cliponaxis=False
            ))

            max_x = max(1.0, cop_u, cfi_u, x_recebimento)  # usa recebimento como referência do topo
            fig_regua.update_layout(
                height=320,
                barmode="overlay",
                xaxis=dict(title="Dias (a partir do pagamento ao fornecedor)", range=[0, max_x * 1.10]),
                yaxis=dict(title="", showticklabels=True),
                margin=dict(l=10, r=10, t=80, b=10),
                legend=dict(orientation="h", yanchor="bottom", y=1.15, xanchor="left", x=0)
            )

            fig_regua.update_yaxes(automargin=True)

            st.plotly_chart(fig_regua, use_container_width=True)

            st.divider()


            # =========================================================
            # 2) EVOLUÇÃO (selecionável)
            # =========================================================
            st.markdown("#### Evolução — ciclos e prazos (dias)")

            opcoes = ["PMR (dias)", "PME (dias)", "PMP (dias)", "Ciclo Operacional", "Ciclo Financeiro"]
            selecionados = st.multiselect(
                "Selecione séries para plotar",
                options=opcoes,
                default=["PMR (dias)", "PME (dias)", "PMP (dias)", "Ciclo Operacional", "Ciclo Financeiro"]
            )

            fig_evo = go.Figure()
            for nome in selecionados:
                y = [float(df_ciclos.loc[nome, a]) if pd.notna(df_ciclos.loc[nome, a]) else np.nan for a in anos_ok]
                fig_evo.add_trace(go.Scatter(x=anos_ok, y=y, mode="lines+markers", name=nome))

            fig_evo.update_layout(
                height=420,
                xaxis_title="Período",
                yaxis_title="Dias",
                margin=dict(l=10, r=10, t=10, b=10),
                legend_title="Séries"
            )
            st.plotly_chart(fig_evo, use_container_width=True)

            st.divider()

            # =========================================================
            # 3) ALAVANCAGENS (Operacional, Financeira e Total)
            # =========================================================
            st.markdown("### ⚙️ Alavancagens — Operacional (DOL), Financeira (DFL) e Total (DTL)")

            # Séries necessárias (já existem na sua DRE)
            ebit = get_serie(dre_df, "Lucro Operacional - EBIT")
            lair = get_serie(dre_df, "Lucro Antes do IR")  # EBT
            lucroliq = get_serie(dre_df, "Lucro Líquido")

            def pct_change_series(s: pd.Series):
                out = {}
                for j, a in enumerate(anos_ok):
                    if j == 0:
                        out[a] = np.nan
                        continue
                    a_prev = anos_ok[j-1]
                    v0 = float(s[a_prev]); v1 = float(s[a])
                    if v0 == 0:
                        out[a] = np.nan
                    else:
                        out[a] = (v1 - v0) / v0
                return pd.Series(out)

            # %Δ
            d_rev = pct_change_series(receita)
            d_ebit = pct_change_series(ebit)
            d_lair = pct_change_series(lair)
            d_ll = pct_change_series(lucroliq)

            # DOL = %ΔEBIT / %ΔReceita
            dol = pd.Series({a: safe_div_scalar(d_ebit[a], d_rev[a]) for a in anos_ok})
            # DFL = %ΔLAIR / %ΔEBIT  (proxy clássico: EBIT -> EBT)
            dfl = pd.Series({a: safe_div_scalar(d_lair[a], d_ebit[a]) for a in anos_ok})
            # DTL = DOL * DFL (ou %ΔLL / %ΔReceita; aqui deixo o padrão multiplicativo)
            dtl = dol * dfl

            df_alav = pd.DataFrame({
                "DOL (Operacional)": dol,
                "DFL (Financeira)": dfl,
                "DTL (Total)": dtl
            }).T
            df_alav = df_alav[anos_ok]
            df_alav.index.name = "Indicador"

            # Tabela (compacta)
            st.dataframe(
                df_alav.style.format({a: "{:,.2f}" for a in anos_ok}),
                use_container_width=True,
                height=min(280, 40 + 32 * (len(df_alav) + 2))
            )

            # Gráfico — evolução das alavancagens
            fig_alav = go.Figure()
            for nome in df_alav.index:
                y = [float(df_alav.loc[nome, a]) if pd.notna(df_alav.loc[nome, a]) else np.nan for a in anos_ok]
                fig_alav.add_trace(go.Scatter(x=anos_ok, y=y, mode="lines+markers", name=nome))

            fig_alav.update_layout(
                height=380,
                xaxis_title="Período",
                yaxis_title="Multiplicador (x)",
                margin=dict(l=10, r=10, t=10, b=10),
                legend_title="Alavancagens"
            )
            st.plotly_chart(fig_alav, use_container_width=True)


        # =================================================
        # SUBABA — TESOURARIA (Fleuriet)
        # =================================================
        with sub_tes:
            st.markdown("### 🏦 Tesouraria — IOG, CPL e Saldo de Tesouraria")

            dre_df = st.session_state.get("dre_df")
            bp_df  = st.session_state.get("balanco_df")

            if dre_df is None or bp_df is None or dre_df.empty or bp_df.empty:
                st.warning("Preencha DRE e Balanço na aba 'Banco de Dados' para habilitar Tesouraria.")
            else:
                # Helpers compatíveis com símbolos em Conta
                def _conta_col(df):
                    return df["Conta"].astype(str).map(conta_limpa)

                def get_serie(df, conta):
                    s = _conta_col(df)
                    mask = (s == conta)
                    if not mask.any():
                        return pd.Series({a: 0.0 for a in anos})
                    out = df.loc[mask, anos].iloc[0]
                    return pd.to_numeric(out, errors="coerce").fillna(0.0)

                # Séries do BP
                cr   = get_serie(bp_df, "Contas a Receber")
                est  = get_serie(bp_df, "Estoques")
                adi  = get_serie(bp_df, "Adiantamentos")

                forn = get_serie(bp_df, "Fornecedores")
                sal  = get_serie(bp_df, "Salários")
                imp  = get_serie(bp_df, "Impostos e Encargos Sociais")

                anc  = get_serie(bp_df, "Ativo Não Circulante")
                pnc  = get_serie(bp_df, "Passivo Não Circulante")
                pl   = get_serie(bp_df, "Patrimônio Líquido")

                # Série da DRE
                vendas = get_serie(dre_df, "Receita Líquida")

                # Definições (como você passou)
                acc = cr + est + adi
                pcc = forn + sal + imp
                iog = acc - pcc

                cpl = (pnc + pl) - anc
                saldo_tes = iog - cpl  # seu padrão

                # -----------------------------
                # Tabela invertida (linhas=variáveis, colunas=anos)
                # -----------------------------
                df_tes = pd.DataFrame({
                    "Vendas (Receita Líquida)": vendas,
                    "ACC (CR + Estoques + Adiant.)": acc,
                    "PCC (Forn + Sal + Imp)": pcc,
                    "IOG (ACC - PCC)": iog,
                    "CPL ((PNC + PL) - ANC)": cpl,
                    "Saldo de Tesouraria (IOG - CPL)": saldo_tes,
                }).T
                df_tes.columns = anos

                # Mostra somente anos preenchidos (evita tabela “toda zero”)
                anos_plot = []
                for a in anos:
                    col = pd.to_numeric(df_tes[a], errors="coerce").fillna(0.0)
                    if float(col.abs().sum()) != 0.0:
                        anos_plot.append(a)
                if not anos_plot:
                    anos_plot = anos[:]

                df_tes_show = df_tes[anos_plot]

                st.dataframe(
                    df_tes_show.style.format({a: "R$ {:,.0f}" for a in anos_plot}),
                    use_container_width=True,
                    height=min(520, 40 + 32 * (len(df_tes_show) + 2))
                )

                st.divider()

                # -----------------------------
                # Gráfico (evolução)
                # -----------------------------
                st.markdown("#### Evolução — selecione as séries")

                opcoes = list(df_tes_show.index)
                default_sel = [
                    "Vendas (Receita Líquida)",
                    "IOG (ACC - PCC)",
                    "CPL ((PNC + PL) - ANC)",
                    "Saldo de Tesouraria (IOG - CPL)"
                ]
                default_sel = [x for x in default_sel if x in opcoes]

                sel = st.multiselect(
                    "Séries",
                    options=opcoes,
                    default=default_sel
                )

                normalizar = st.checkbox("Normalizar (base 100 no primeiro ano com valor)", value=False)

                def _norm_row(row: pd.Series) -> pd.Series:
                    if not normalizar:
                        return row
                    base = None
                    for a in anos_plot:
                        v = float(row[a])
                        if v != 0.0 and np.isfinite(v):
                            base = v
                            break
                    if base in (None, 0.0) or not np.isfinite(base):
                        return row * 0.0
                    return (row / base) * 100.0

                fig = go.Figure()
                for nome in sel:
                    yrow = df_tes_show.loc[nome, anos_plot].astype(float)
                    yrow = _norm_row(yrow)
                    fig.add_trace(go.Scatter(
                        x=anos_plot,
                        y=[float(yrow[a]) if pd.notna(yrow[a]) else np.nan for a in anos_plot],
                        mode="lines+markers",
                        name=nome
                    ))

                fig.update_layout(
                    height=520,
                    xaxis_title="Período",
                    yaxis_title="Base 100" if normalizar else "R$",
                    legend_title="Séries",
                    margin=dict(l=10, r=10, t=10, b=10)
                )

                st.plotly_chart(fig, use_container_width=True)


        # =================================================
        # SUBABA — WACC & Valor
        # =================================================
        with sub_wacc:
            st.markdown("## 💰 Custo Médio Ponderado de Capital (WACC)")

            # -----------------------------
            # Inputs (parâmetros)
            # -----------------------------
            c1, c2, c3, c4 = st.columns(4)
            with c1:
                rf = st.number_input("Taxa livre de risco (Rf) % a.a.", value=10.0, step=0.25) / 100.0
            with c2:
                mrp = st.number_input("Prêmio de risco de mercado % a.a.", value=6.0, step=0.25) / 100.0
            with c3:
                beta = st.number_input("Beta da empresa", value=1.0, step=0.05)
            with c4:
                tax = st.number_input("Alíquota de IR/CSLL %", value=34.0, step=1.0) / 100.0

            kd = st.number_input("Custo médio da dívida (Kd) % a.a.", value=12.0, step=0.25) / 100.0

            st.divider()

            # -----------------------------
            # Séries base (BP / DRE)
            # -----------------------------
            # Dívida bruta (proxy = empréstimos CP + LP) e PL
            div_cp = get_serie(bp_df, "Empréstimos e Financiamentos (CP)")
            div_lp = get_serie(bp_df, "Empréstimos e Financiamentos (LP)")
            debt = div_cp + div_lp

            equity = get_serie(bp_df, "Patrimônio Líquido")

            # ROIC (recalcula aqui para não depender de outra subaba)
            ebit = get_serie(dre_df, "Lucro Operacional - EBIT")

            # Capital investido (proxy coerente com seu Fleuriet)
            cr   = get_serie(bp_df, "Contas a Receber")
            est  = get_serie(bp_df, "Estoques")
            adi  = get_serie(bp_df, "Adiantamentos")
            forn = get_serie(bp_df, "Fornecedores")
            sal  = get_serie(bp_df, "Salários")
            imp  = get_serie(bp_df, "Impostos e Encargos Sociais")
            anc  = get_serie(bp_df, "Ativo Não Circulante")

            acc = cr + est + adi
            pcc = forn + sal + imp
            iog = acc - pcc

            # NOPAT e ROIC
            nopat = ebit * (1.0 - tax)
            cap_inv = iog + anc
            roic = safe_div(nopat, cap_inv) * 100.0  # em %

            # -----------------------------
            # WACC por ano
            # -----------------------------
            ke = (rf + beta * mrp) * 100.0           # Ke em %
            kd_after = (kd * (1.0 - tax)) * 100.0    # Kd pós-IR em %

            # pesos por ano (como Series indexada por "Ano i")
            w_d = pd.Series(index=anos_ok, dtype=float)
            w_e = pd.Series(index=anos_ok, dtype=float)
            wacc = pd.Series(index=anos_ok, dtype=float)

            for a in anos_ok:
                d = float(debt[a])
                e = float(equity[a])
                tot = d + e

                if tot == 0:
                    w_d[a] = np.nan
                    w_e[a] = np.nan
                    wacc[a] = np.nan
                else:
                    w_d[a] = (d / tot) * 100.0
                    w_e[a] = (e / tot) * 100.0
                    wacc[a] = ke * (w_e[a] / 100.0) + kd_after * (w_d[a] / 100.0)


            # -----------------------------
            # Tabela (invertida: anos como colunas)
            # -----------------------------
            # Monta DF com anos nas linhas e depois transpõe
            df_wacc = pd.DataFrame(index=anos_ok)
            df_wacc["Dívida Bruta"] = [float(debt[a]) for a in anos_ok]
            df_wacc["Patrimônio Líquido"] = [float(equity[a]) for a in anos_ok]
            df_wacc["Peso Dívida (%)"] = [float(w_d[a]) if pd.notna(w_d[a]) else np.nan for a in anos_ok]
            df_wacc["Peso PL (%)"] = [float(w_e[a]) if pd.notna(w_e[a]) else np.nan for a in anos_ok]
            df_wacc["Ke (%)"] = ke
            df_wacc["Kd pós-IR (%)"] = kd_after
            df_wacc["WACC (%)"] = [float(wacc[a]) if pd.notna(wacc[a]) else np.nan for a in anos_ok]
            
            # --- Garantir ROIC como Series indexada por anos_ok (evita IndexError quando roic vira ndarray) ---
            if isinstance(roic, pd.Series):
                roic_s = roic.reindex(anos_ok).astype(float)
            else:
                # tenta interpretar como lista/ndarray na mesma ordem de anos_ok
                try:
                    roic_s = pd.Series(list(roic), index=anos_ok, dtype=float)
                except Exception:
                    roic_s = pd.Series({a: np.nan for a in anos_ok}, dtype=float)

            df_wacc["ROIC (%)"] = [float(roic_s[a]) if pd.notna(roic_s[a]) else np.nan for a in anos_ok]
            df_wacc["Spread (ROIC - WACC) p.p."] = df_wacc["ROIC (%)"] - df_wacc["WACC (%)"]

            # Inverte: linhas = métricas, colunas = anos
            df_wacc_t = df_wacc.T
            df_wacc_t.columns = anos_ok

            # Formatação
            fmt = {}
            for a in anos_ok:
                fmt[a] = lambda v: "" if (v is None or (isinstance(v, float) and np.isnan(v))) else f"{v:,.2f}"

            # Ajustes por métrica
            def fmt_money(v):
                if v is None or (isinstance(v, float) and np.isnan(v)): return ""
                return f"R$ {v:,.0f}"

            def fmt_pct(v):
                if v is None or (isinstance(v, float) and np.isnan(v)): return ""
                return f"{v:,.2f}%"

            def fmt_pp(v):
                if v is None or (isinstance(v, float) and np.isnan(v)): return ""
                return f"{v:+.2f} p.p."

            format_dict = {}
            for idx in df_wacc_t.index:
                if idx in ["Dívida Bruta", "Patrimônio Líquido"]:
                    format_dict[idx] = fmt_money
                elif "p.p." in idx:
                    format_dict[idx] = fmt_pp
                else:
                    format_dict[idx] = fmt_pct if ("%" in idx) else (lambda v: f"{v:,.2f}" if pd.notna(v) else "")

            # Altura exata (evita “linhas vazias” visuais)
            altura = min(520, 40 + 32 * (len(df_wacc_t) + 1))

            st.dataframe(
                df_wacc_t.style.format(format_dict, subset=pd.IndexSlice[:, anos_ok]),
                use_container_width=True,
                height=altura
            )

            st.divider()

            # -----------------------------
            # Gráfico (anos no eixo X)
            # -----------------------------
            st.markdown("### 📈 WACC vs ROIC (e Spread)")

            x = anos_ok
            y_wacc = [float(df_wacc.loc[a, "WACC (%)"]) if pd.notna(df_wacc.loc[a, "WACC (%)"]) else np.nan for a in x]
            y_roic = [float(df_wacc.loc[a, "ROIC (%)"]) if pd.notna(df_wacc.loc[a, "ROIC (%)"]) else np.nan for a in x]
            y_spread = [float(df_wacc.loc[a, "Spread (ROIC - WACC) p.p."]) if pd.notna(df_wacc.loc[a, "Spread (ROIC - WACC) p.p."]) else np.nan for a in x]

            fig = go.Figure()
            fig.add_trace(go.Scatter(x=x, y=y_wacc, mode="lines+markers", name="WACC (%)"))
            fig.add_trace(go.Scatter(x=x, y=y_roic, mode="lines+markers", name="ROIC (%)"))
            fig.add_trace(go.Scatter(x=x, y=y_spread, mode="lines+markers", name="Spread (p.p.)"))

            fig.update_layout(
                height=480,
                xaxis_title="Período",
                yaxis_title="%",
                legend_title="Séries",
                margin=dict(l=10, r=10, t=10, b=10)
            )
            st.plotly_chart(fig, use_container_width=True)

            st.divider()

            # -----------------------------
            # Criação de valor (mensagem)
            # -----------------------------
            # Se ROIC estiver todo NaN (ex.: capital investido zero), avisa corretamente.
            if np.all(np.isnan(np.array(y_roic, dtype=float))):
                st.info("ROIC não disponível para comparação (capital investido ficou 0 ou dados insuficientes nos itens do BP para IOG/ANC).")
            else:
                # Usa último ano com valores válidos
                last_valid = None
                for a in reversed(x):
                    r = df_wacc.loc[a, "ROIC (%)"]
                    w = df_wacc.loc[a, "WACC (%)"]
                    if pd.notna(r) and pd.notna(w):
                        last_valid = a
                        break

                if last_valid is None:
                    st.info("ROIC/WACC não disponíveis no mesmo período para comparação.")
                else:
                    spread_last = float(df_wacc.loc[last_valid, "Spread (ROIC - WACC) p.p."])
                    if spread_last >= 0:
                        st.success(f"No **{last_valid}**, a empresa cria valor: **ROIC - WACC = {spread_last:+.2f} p.p.**")
                    else:
                        st.warning(f"No **{last_valid}**, a empresa destrói valor: **ROIC - WACC = {spread_last:+.2f} p.p.**")


# =========================================================
# TAB 3 — MATRIZ DO CAIXA (OPERACIONAL) — QUINZENAL
# =========================================================
with tab3:
    st.subheader("🧩 Matriz do Caixa — Exposição Máxima de Capital de Giro (Ciclo Operacional)")
    st.caption("Matriz quinzenal baseada em CMV como origem do ciclo. Considera ativos e passivos cíclicos; salários+impostos pagos 1x/mês (a cada 2 quinzenas).")

    dre_df = st.session_state.get("dre_df")
    bp_df  = st.session_state.get("balanco_df")

    if dre_df is None or bp_df is None or dre_df.empty or bp_df.empty:
        st.warning("Preencha DRE e Balanço na aba 'Banco de Dados' para habilitar a Matriz do Caixa.")
    else:
        # -----------------------------
        # Helpers locais (não altera seus globais)
        # -----------------------------
        def _conta_col(df):
            return df["Conta"].astype(str).map(conta_limpa)

        def get_serie_local(df, conta):
            s = _conta_col(df)
            mask = (s == conta)
            if not mask.any():
                return pd.Series({a: 0.0 for a in anos})
            out = df.loc[mask, anos].iloc[0]
            return pd.to_numeric(out, errors="coerce").fillna(0.0)

        def safe_div_scalar(n, d):
            try:
                n = float(n); d = float(d)
                return np.nan if d == 0 else n/d
            except Exception:
                return np.nan

        def ceil_quinzena(dias):
            # converte dias -> número de quinzenas (arredonda para cima)
            try:
                dias = float(dias)
            except Exception:
                dias = 0.0
            return int(np.ceil(max(0.0, dias) / 15.0))

        # -----------------------------
        # Seleção do ano de referência
        # -----------------------------
        # Usa anos_ok (já calculado no tab2). Se não existir por algum motivo, usa anos completo.
        anos_ok_local = anos_ok if "anos_ok" in locals() and len(anos_ok) > 0 else anos[:]
        ano_ref = st.selectbox("Período de referência", options=anos_ok_local, index=len(anos_ok_local)-1)

        # -----------------------------
        # Inputs dos prazos (dias)
        # (Se você já tem PMR/PME/PMP calculados e armazenados, pode ligar aqui.
        #  Por enquanto, input explícito deixa a matriz independente e previsível.)
        # -----------------------------
        c_prazos1, c_prazos2, c_prazos3 = st.columns(3)
        with c_prazos1:
            pmr_d = st.number_input("PMR (dias)", value=35.0, step=1.0)
        with c_prazos2:
            pme_d = st.number_input("PME (dias)", value=96.0, step=1.0)
        with c_prazos3:
            pmp_d = st.number_input("PMP (dias)", value=57.0, step=1.0)

        pmr_q = ceil_quinzena(pmr_d)
        pme_q = ceil_quinzena(pme_d)
        pmp_q = ceil_quinzena(pmp_d)

        ciclo_oper_q = pmr_q + pme_q
        ciclo_fin_q  = max(0, ciclo_oper_q - pmp_q)  # apenas referência; matriz é operacional

        st.caption(f"Conversão para quinzena (≈15 dias): PME={pme_q}q, PMR={pmr_q}q, PMP={pmp_q}q | Ciclo Operacional={ciclo_oper_q}q | Ciclo Financeiro={ciclo_fin_q}q")

        st.divider()

        # -----------------------------
        # Séries contábeis (ano_ref)
        # -----------------------------
        receita = float(get_serie_local(dre_df, "Receita Líquida").get(ano_ref, 0.0))
        cmv_abs = float(get_serie_local(dre_df, "CMV, CPV ou CSP").abs().get(ano_ref, 0.0))

        cr   = float(get_serie_local(bp_df, "Contas a Receber").get(ano_ref, 0.0))
        est  = float(get_serie_local(bp_df, "Estoques").get(ano_ref, 0.0))
        adi  = float(get_serie_local(bp_df, "Adiantamentos").get(ano_ref, 0.0))

        forn = float(get_serie_local(bp_df, "Fornecedores").get(ano_ref, 0.0))
        sal  = float(get_serie_local(bp_df, "Salários").get(ano_ref, 0.0))
        imp  = float(get_serie_local(bp_df, "Impostos e Encargos Sociais").get(ano_ref, 0.0))

        # Ativos e passivos cíclicos (como você definiu antes)
        incluir_adi = st.checkbox("Incluir Adiantamentos no ACC", value=True)
        acc_val = cr + est + (adi if incluir_adi else 0.0)
        pcc_val = forn + sal + imp
        iog_val = acc_val - pcc_val

        c1, c2, c3 = st.columns(3)
        c1.metric("ACC (R$)", f"R$ {acc_val:,.0f}".replace(",", "."))
        c2.metric("PCC (R$)", f"R$ {pcc_val:,.0f}".replace(",", "."))
        c3.metric("IOG (R$)", f"R$ {iog_val:,.0f}".replace(",", "."))

        st.divider()

        # -----------------------------
        # Matriz Operacional (quinzenal) — CMV como origem
        # -----------------------------
        # Ideia:
        # - cada "coorte" de compra (CMV) nasce na quinzena t
        # - fica "presa" em estoque por pme_q
        # - depois vira CR por pmr_q
        # - o financiamento por fornecedores atua até pmp_q
        # - salários+impostos: descarga mensal (a cada 2 quinzenas), aproximada como parcelas iguais no horizonte

        # Horizonte mínimo para visualizar bem (1 ciclo operacional + folga)
        horizon_q = max(8, ciclo_oper_q + 6)  # você pode ajustar
        quinz = list(range(1, horizon_q + 1))

        # Volume de CMV por quinzena (origem)
        # Ano contábil -> 24 quinzenas/ano. Se faltarem dados, cai para 0.
        cmv_por_q = cmv_abs / 24.0 if cmv_abs else 0.0

        # Percentuais de entrada (sem parcelamento) — você comentou que isso é interessante
        # Aqui aplicamos na compra (fornecedor) e na venda (cliente)
        cc1, cc2 = st.columns(2)
        with cc1:
            pct_entrada_compra = st.slider("% entrada na compra (fornecedor)", 0.0, 100.0, 0.0, 1.0) / 100.0
        with cc2:
            pct_entrada_venda = st.slider("% entrada na venda (cliente)", 0.0, 100.0, 0.0, 1.0) / 100.0

        # Para manter coerência, se há entrada na venda, reduz o CR "final" associado às vendas futuras.
        # (modelo simples e transparente)

        # Componentes unitários por coorte (baseados na estrutura do BP)
        # Proporção do ACC em relação ao "nível operacional" (cmv/receita ajuda a calibrar, mas pode ser ruim se receita=0)
        # Aqui usamos o próprio BP como tamanho-alvo; a matriz é exposição, então é razoável distribuir o ACC/PCC pelo ciclo.
        # Parte do ACC que é estoque vs CR (usamos peso do BP)
        acc_component = est + cr + (adi if incluir_adi else 0.0)
        w_est = safe_div_scalar(est, acc_component) if acc_component else np.nan
        w_cr  = safe_div_scalar(cr,  acc_component) if acc_component else np.nan
        w_adi = safe_div_scalar((adi if incluir_adi else 0.0), acc_component) if acc_component else np.nan

        if not np.isfinite(w_est): w_est = 0.5
        if not np.isfinite(w_cr):  w_cr  = 0.5
        if not np.isfinite(w_adi): w_adi = 0.0

        # Passivos cíclicos: fornecedores vs (sal+imp)
        pcc_component = forn + sal + imp
        w_forn = safe_div_scalar(forn, pcc_component) if pcc_component else np.nan
        w_folha = safe_div_scalar((sal + imp), pcc_component) if pcc_component else np.nan

        if not np.isfinite(w_forn):  w_forn = 0.6
        if not np.isfinite(w_folha): w_folha = 0.4

        # Matrizes: linhas = coortes (1..horizon), colunas = quinzenas (1..horizon)
        # Valores: exposição incremental da coorte na quinzena
        M = np.zeros((horizon_q, horizon_q), dtype=float)

        for t0 in range(1, horizon_q + 1):
            # Tamanho do "lote" operacional por coorte
            lote = cmv_por_q

            # 1) Estoque (usa caixa): do t0 até t0+pme_q-1
            # (parcela de estoque + adiantamentos)
            est_val = lote * w_est
            adi_val = lote * w_adi
            for t in range(t0, min(horizon_q, t0 + pme_q - 1) + 1):
                M[t0-1, t-1] += (est_val + adi_val)

            # 2) CR (usa caixa): após PME, entra em CR por PMR
            cr_val = lote * w_cr
            # entrada na venda reduz CR final (modelo simples)
            cr_val = cr_val * (1.0 - pct_entrada_venda)

            cr_ini = t0 + pme_q
            cr_fim = t0 + pme_q + pmr_q - 1
            for t in range(cr_ini, min(horizon_q, cr_fim) + 1):
                M[t0-1, t-1] += cr_val

            # 3) Fornecedores (financia): reduz exposição até PMP
            # entrada na compra antecipa pagamento (reduz fornecedor financiando)
            forn_val = lote * w_forn
            # parte paga à vista na compra => não financia, então não entra como "redução" ao longo
            forn_val_fin = forn_val * (1.0 - pct_entrada_compra)

            forn_ini = t0
            forn_fim = t0 + pmp_q - 1
            for t in range(forn_ini, min(horizon_q, forn_fim) + 1):
                M[t0-1, t-1] -= forn_val_fin

            # 4) Salários + Impostos (financia parcialmente? na prática é obrigação, então "reduz caixa" no pagamento)
            # Para matriz de exposição, a forma mais clara é:
            # - não reduzir exposição "todo dia", e sim colocar "descargas" mensais.
            # Aqui vamos modelar como pagamentos mensais (a cada 2 quinzenas), distribuídos pelo horizonte,
            # proporcional ao lote e ao peso folha.
            folha_val = lote * w_folha

            # Pagamento mensal: ocorre em quinzena par (2,4,6,...)
            # Começa 1 mês após origem (t0+1) para evitar pagar antes de existir operação.
            # Ajuste simples e transparente.
            for t in range(2, horizon_q + 1, 2):
                if t >= t0 + 1:
                    # descarga: aumenta exposição (paga => precisa de caixa)
                    # para manter consistência com "PCC" como fonte, aqui tratamos folha como "passivo cíclico"
                    # e portanto NEGATIVA enquanto não paga. Como não estamos acumulando passivo de folha ao longo,
                    # fazemos a aproximação: parte da folha reduz exposição no intervalo e volta no pagamento.
                    # Simplificação: reduz por 2 quinzenas e volta no t par.
                    # (Se quiser, depois refinamos com calendário específico.)
                    t_ini = max(t-1, t0)
                    t_fim = min(t, horizon_q)
                    # "financiamento" no intervalo
                    M[t0-1, t_ini-1] -= folha_val
                    # pagamento no t (reverte)
                    M[t0-1, t_fim-1] += folha_val

        # Exposição total por quinzena (soma das coortes)
        expos_q = M.sum(axis=0)
        expo_max = float(np.nanmax(expos_q))
        t_pico = int(np.nanargmax(expos_q) + 1)

        st.markdown("### Debug — Sensibilidade dos parâmetros")

        st.write("pct_entrada_compra =", pct_entrada_compra, "| pct_entrada_venda =", pct_entrada_venda)

        # Mostra estatísticas básicas da matriz
        st.write("M min/max:", float(np.nanmin(M)), float(np.nanmax(M)))
        st.write("Soma total de M (deveria tender a 0 se fechou o ciclo):", float(np.nansum(M)))

        # fluxo e saldo
        fluxo_q = np.nansum(M, axis=0)
        saldo_acum = np.cumsum(fluxo_q)

        st.write("Fluxo por quinzena (primeiras 8):", [float(v) for v in fluxo_q[:8]])
        st.write("Saldo acumulado (primeiras 8):", [float(v) for v in saldo_acum[:8]])
        st.write("Pico uso (min saldo):", float(np.nanmin(saldo_acum)))


    
        # -----------------------------
        # Visualizações
        # -----------------------------
        st.markdown("### Heatmap — Exposição por coorte (quinzenas)")
        st.caption("Linhas = coortes de CMV (início da compra). Colunas = quinzenas. Valores positivos indicam necessidade (uso); negativos indicam financiamento operacional.")

        # ===== AJUSTE DE VISUAL (cores + zero + escala simétrica) =====
        colorscale_divergente = [
            [0.00, "#2166ac"],  # azul escuro (negativo forte = financiamento)
            [0.45, "#d1e5f0"],  # azul claro
            [0.50, "#ffffff"],  # branco (zero)
            [0.55, "#fddbc7"],  # vermelho claro
            [1.00, "#b2182b"],  # vermelho escuro (positivo forte = necessidade)
        ]

        # garante escala simétrica em torno do zero (melhora MUITO a leitura)
        M_np = np.asarray(M, dtype=float)
        absmax = float(np.nanmax(np.abs(M_np))) if np.isfinite(np.nanmax(np.abs(M_np))) else 1.0
        zmin, zmax = -absmax, absmax

        x_labels = [f"Q{t}" for t in quinz]
        y_labels = [f"Coorte Q{t}" for t in quinz]

        fig_hm = go.Figure(data=go.Heatmap(
            z=M,
            x=x_labels,
            y=y_labels,
            colorscale=colorscale_divergente,
            zmid=0,           # ancora o zero no branco
            zmin=zmin,        # escala simétrica
            zmax=zmax,
            colorbar=dict(
                title="Exposição (R$)",
                tickformat=",.0f"
            ),
            hovertemplate=(
                "<b>%{y}</b> → %{x}<br>"
                "Exposição: <b>R$ %{z:,.0f}</b><br>"
                "<span style='font-size:12px'>"
                "Positivo = uso (necessidade de capital)<br>"
                "Negativo = fonte (financiamento operacional)"
                "</span>"
                "<extra></extra>"
            )
        ))

        # melhora margens/topo (evita sensação de “apertado”)
        fig_hm.update_layout(
            height=560,
            margin=dict(l=10, r=10, t=30, b=10)
        )

        st.plotly_chart(fig_hm, use_container_width=True)

        st.divider()

        # -----------------------------
        # Curva — SALDO ACUMULADO (estoque de caixa necessário)
        # -----------------------------
        st.markdown("### Curva — Saldo acumulado do ciclo (identifica o pico de uso do caixa)")
        st.caption("Aqui o gráfico mostra o *estoque* de caixa ao longo das quinzenas (cumulativo). O pico de necessidade é o menor saldo (mais negativo).")

        # 1) fluxo por quinzena = soma das coortes (colunas da matriz)
        fluxo_q = np.nansum(M, axis=0)  # shape (len(quinz),)

        # 2) saldo acumulado (começa em 0)
        saldo_acum = np.cumsum(fluxo_q)

        # 3) pico de uso de caixa = menor saldo (mais negativo)
        t_min = int(np.nanargmin(saldo_acum))  # índice
        saldo_min = float(saldo_acum[t_min])
        uso_max = -saldo_min if saldo_min < 0 else 0.0
        q_pico = quinz[t_min]

        fig_saldo = go.Figure()

        # linha do saldo acumulado
        fig_saldo.add_trace(go.Scatter(
            x=[f"Q{t}" for t in quinz],
            y=[float(v) for v in saldo_acum],
            mode="lines+markers",
            name="Saldo acumulado (R$)"
        ))

        # destaca o ponto de pico (mínimo)
        fig_saldo.add_trace(go.Scatter(
            x=[f"Q{q_pico}"],
            y=[saldo_min],
            mode="markers+text",
            text=[f"Pico uso: R$ {uso_max:,.0f}".replace(",", ".")],
            textposition="top center",
            marker=dict(size=10),
            showlegend=False
        ))

        # linha zero para referência
        fig_saldo.add_hline(y=0, line_width=1)

        fig_saldo.update_layout(
            height=420,
            margin=dict(l=10, r=10, t=10, b=10),
            xaxis_title="Quinzena",
            yaxis_title="R$ (saldo acumulado)"
        )

        st.plotly_chart(fig_saldo, use_container_width=True)

        st.success(
            f"Pico de necessidade de caixa no ciclo: **R$ {uso_max:,.0f}** na **Q{q_pico}** (saldo mínimo = {saldo_min:,.0f})."
            .replace(",", ".")
        )


        st.divider()

        st.markdown("### Parâmetros usados (transparência do modelo)")
        df_params = pd.DataFrame({
            "Parâmetro": ["PME (q)", "PMR (q)", "PMP (q)", "Ciclo Operacional (q)", "CMV por quinzena", "% entrada compra", "% entrada venda"],
            "Valor": [pme_q, pmr_q, pmp_q, ciclo_oper_q, cmv_por_q, pct_entrada_compra*100, pct_entrada_venda*100]
        })
        st.dataframe(df_params, use_container_width=True, hide_index=True)


