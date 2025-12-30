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
tab1, tab2 = st.tabs(["📥 Banco de Dados", "📈 Análises Financeiras"])

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

        # Editor (NUNCA converter antes)
        dre_raw = st.data_editor(
            st.session_state["dre_raw"],
            disabled=["Conta"],
            num_rows="fixed",
            use_container_width=True,
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

        bp_raw = st.data_editor(
            st.session_state["bp_raw"],
            disabled=["Conta"],
            num_rows="fixed",
            use_container_width=True,
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
            return np.where(np.asarray(d) == 0, np.nan, np.asarray(n) / np.asarray(d))

        # Anos efetivamente preenchidos (algum valor diferente de zero)
        def anos_preenchidos(df):
            cols = []
            for a in anos:
                col = pd.to_numeric(df[a], errors="coerce").fillna(0.0)
                if float(col.abs().sum()) != 0.0:
                    cols.append(a)
            return cols

        anos_ok = sorted(set(anos_preenchidos(dre_df)) | set(anos_preenchidos(bp_df)), key=lambda x: int(x.split()[-1]))
        if len(anos_ok) == 0:
            anos_ok = anos[:]  # fallback

        # -------------------------------------------------
        # Subabas
        # -------------------------------------------------
        sub_avah, sub_ciclos, sub_tes = st.tabs(["📊 Vertical & Horizontal", "⏱️ PMR • PME • PMP", "🏦 Tesouraria"])

        # =================================================
        # SUBABA 1 — Vertical & Horizontal
        # =================================================
        with sub_avah:
            st.markdown("### 📊 Análise Vertical e Horizontal")

            alvo = st.selectbox("Escolha a demonstração", ["DRE", "Balanço Patrimonial"], index=0)

            if alvo == "DRE":
                df_base = dre_df.copy()
                # Base da vertical: Receita Líquida
                base_conta = "Receita Líquida"
                base_nome = "Receita Líquida"
            else:
                df_base = bp_df.copy()
                # Base da vertical: Ativo Circulante + Ativo Não Circulante (proxy do Ativo Total na sua estrutura)
                # Como você não tem "Ativo Total", usamos "Ativo Circulante" + "Ativo Não Circulante"
                base_conta = None
                base_nome = "Ativo Total (AC + ANC)"

            # Normaliza colunas numéricas
            for a in anos:
                df_base[a] = pd.to_numeric(df_base[a], errors="coerce").fillna(0.0)

            # Coluna lógica
            df_base["_Conta_Limpa"] = df_base["Conta"].astype(str).map(conta_limpa)

            # -------- Vertical (%)
            if alvo == "DRE":
                base = get_serie(df_base, base_conta)
            else:
                base = get_serie(df_base, "Ativo Circulante") + get_serie(df_base, "Ativo Não Circulante")

            df_vert = df_base[["Conta"] + anos_ok].copy()
            for a in anos_ok:
                df_vert[a] = safe_div(df_vert[a].values, float(base[a])) * 100.0

            # -------- Horizontal (% var e var abs)
            df_habs = df_base[["Conta"] + anos_ok].copy()
            df_hpct = df_base[["Conta"] + anos_ok].copy()

            # var abs e % vs ano anterior (Ano i vs Ano i-1)
            for j in range(1, len(anos_ok)):
                a_now = anos_ok[j]
                a_prev = anos_ok[j-1]
                abs_var = df_base[a_now] - df_base[a_prev]
                pct_var = safe_div(abs_var.values, df_base[a_prev].values) * 100.0
                df_habs[a_now] = abs_var
                df_hpct[a_now] = pct_var

            # primeiro ano não tem comparação
            if len(anos_ok) >= 1:
                df_habs[anos_ok[0]] = np.nan
                df_hpct[anos_ok[0]] = np.nan

            c1, c2 = st.columns(2)
            with c1:
                st.markdown(f"#### Vertical (%) — Base: {base_nome}")
                st.dataframe(
                    df_vert.style.format({a: "{:,.2f}%" for a in anos_ok}),
                    use_container_width=True,
                    height=min(1000, 40 + 32 * (len(df_vert) + 2))
                )

            with c2:
                st.markdown("#### Horizontal (Δ % vs período anterior)")
                st.dataframe(
                    df_hpct.style.format({a: "{:,.2f}%" for a in anos_ok}),
                    use_container_width=True,
                    height=min(1000, 40 + 32 * (len(df_hpct) + 2))
                )


        # =================================================
        # SUBABA 2 — PMR / PME / PMP
        # =================================================
        with sub_ciclos:
            st.markdown("### ⏱️ Ciclo de Caixa — PMR, PME, PMP")

            st.caption(
                "Premissas padrão (ajustáveis depois):\n"
                "- **PMR** = Contas a Receber / Receita Líquida × 365\n"
                "- **PME** = Estoques / CMV × 365\n"
                "- **PMP** = Fornecedores / CMV × 365\n"
                "Obs.: CMV é usado em módulo (se estiver negativo no seu modelo)."
            )

            receita = get_serie(dre_df, "Receita Líquida")
            cmv = get_serie(dre_df, "CMV, CPV ou CSP")
            cmv_abs = cmv.abs()

            cr = get_serie(bp_df, "Contas a Receber")
            est = get_serie(bp_df, "Estoques")
            forn = get_serie(bp_df, "Fornecedores")

            df_ciclos = pd.DataFrame({
                "Indicador": ["PMR (dias)", "PME (dias)", "PMP (dias)", "Ciclo Operacional", "Ciclo Financeiro"],
            })

            for a in anos_ok:
                pmr = float(safe_div(cr[a], receita[a]) * 365.0) if receita[a] != 0 else np.nan
                pme = float(safe_div(est[a], cmv_abs[a]) * 365.0) if cmv_abs[a] != 0 else np.nan
                pmp = float(safe_div(forn[a], cmv_abs[a]) * 365.0) if cmv_abs[a] != 0 else np.nan

                ciclo_op = pmr + pme if (pd.notna(pmr) and pd.notna(pme)) else np.nan
                ciclo_fin = ciclo_op - pmp if (pd.notna(ciclo_op) and pd.notna(pmp)) else np.nan

                df_ciclos[a] = [pmr, pme, pmp, ciclo_op, ciclo_fin]

            st.dataframe(
                df_ciclos.style.format({a: "{:,.0f}" for a in anos_ok}),
                use_container_width=True,
                height=min(600, 40 + 35 * (len(df_ciclos) + 2))
            )

            # Cards do último período disponível
            ultimo = anos_ok[-1]
            pmr_u = df_ciclos.loc[df_ciclos["Indicador"] == "PMR (dias)", ultimo].values[0]
            pme_u = df_ciclos.loc[df_ciclos["Indicador"] == "PME (dias)", ultimo].values[0]
            pmp_u = df_ciclos.loc[df_ciclos["Indicador"] == "PMP (dias)", ultimo].values[0]
            cop_u = df_ciclos.loc[df_ciclos["Indicador"] == "Ciclo Operacional", ultimo].values[0]
            cfi_u = df_ciclos.loc[df_ciclos["Indicador"] == "Ciclo Financeiro", ultimo].values[0]

           
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

                # Série da DRE (Vendas)
                vendas = get_serie(dre_df, "Receita Líquida")

                # Cálculos (como você definiu)
                acc = cr + est + adi
                pcc = forn + sal + imp
                iog = acc - pcc

                cpl = (pnc + pl) - anc
                st_saldo = iog - cpl  # conforme seu padrão

                # Monta tabela (linhas variáveis, colunas anos)
                df_tes = pd.DataFrame({
                    "Vendas (Receita Líquida)": vendas,
                    "ACC (CR + Estoques + Adiant.)": acc,
                    "PCC (Forn + Sal + Imp)": pcc,
                    "IOG (ACC - PCC)": iog,
                    "CPL ((PNC + PL) - ANC)": cpl,
                    "Saldo de Tesouraria (IOG - CPL)": st_saldo,
                }).T
                df_tes.columns = anos

                st.dataframe(
                    df_tes.style.format({a: "R$ {:,.0f}" for a in anos}),
                    use_container_width=True,
                    height=min(520, 40 + 32 * (len(df_tes) + 2))
                )

                st.divider()

                # Gráfico (evolução)
                st.markdown("#### Evolução — Vendas, IOG, CPL e Saldo de Tesouraria")

                normalizar = st.checkbox("Normalizar (base 100 no primeiro ano preenchido)", value=False)

                # Detecta anos preenchidos (para não plotar tudo zero)
                anos_plot = []
                for a in anos:
                    col = df_tes[a].astype(float)
                    if float(np.nansum(np.abs(col.values))) != 0.0:
                        anos_plot.append(a)
                if not anos_plot:
                    anos_plot = anos[:]

                # Prepara séries para plot
                def _norm(s: pd.Series) -> pd.Series:
                    if not normalizar:
                        return s
                    # base = primeiro ano com valor != 0
                    base = None
                    for a in anos_plot:
                        v = float(s[a])
                        if v != 0.0:
                            base = v
                            break
                    if base in (None, 0.0):
                        return s * 0.0
                    return (s / base) * 100.0

                x = anos_plot
                vendas_p = _norm(vendas)
                iog_p    = _norm(iog)
                cpl_p    = _norm(cpl)
                st_p     = _norm(st_saldo)

                fig = go.Figure()
                fig.add_trace(go.Scatter(x=x, y=[float(vendas_p[a]) for a in x], mode="lines+markers", name="Vendas"))
                fig.add_trace(go.Scatter(x=x, y=[float(iog_p[a]) for a in x],    mode="lines+markers", name="IOG"))
                fig.add_trace(go.Scatter(x=x, y=[float(cpl_p[a]) for a in x],    mode="lines+markers", name="CPL"))
                fig.add_trace(go.Scatter(x=x, y=[float(st_p[a]) for a in x],     mode="lines+markers", name="Saldo de Tesouraria"))

                fig.update_layout(
                    height=520,
                    xaxis_title="Período",
                    yaxis_title="Base 100" if normalizar else "R$",
                    legend_title="Séries",
                    margin=dict(l=10, r=10, t=10, b=10)
                )

                st.plotly_chart(fig, use_container_width=True)

               