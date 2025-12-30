import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

st.set_page_config(
    page_title="Análise Econômico-Financeira",
    layout="wide"
)

st.title("📊 Análise Econômico-Financeira da Empresa")
st.caption("Preencha os dados do mais antigo para o mais recente")

# =========================================================
# CONSTANTES
# =========================================================
anos = [f"Ano {i}" for i in range(1, 7)]

# =========================================================
# FUNÇÕES AUXILIARES (estilo + formatação segura)
# =========================================================
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

def altura_dataframe(df, max_altura=950, altura_linha=35, altura_header=40, padding=20):
    """
    Calcula uma altura para o st.dataframe sem rolagem interna,
    respeitando um teto (max_altura) para não ficar gigante.
    """
    n = len(df)
    h = altura_header + (n * altura_linha) + padding
    return min(h, max_altura)

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
    """

    df = dre_df.copy()

    # Garantir numérico
    for a in anos:
        df[a] = pd.to_numeric(df[a], errors="coerce").fillna(0.0)

    df_idx = df.set_index("Conta")

    # 1) Normalizar APENAS contas sempre-negativas
    sempre_negativas = [
        "CMV, CPV ou CSP",
        "Despesas de Vendas",
        "Despesas gerais e administrativas",
        "Depreciação & Amortização",
        "Imposto de Renda",
    ]
    for conta in sempre_negativas:
        if conta in df_idx.index:
            for a in anos:
                df_idx.loc[conta, a] = -abs(float(df_idx.loc[conta, a]))

    # Helper: série por ano (0 se não existir)
    def get(conta):
        mask = df_idx.index.map(conta_limpa) == conta
        if mask.any():
            return df_idx.loc[mask, anos].astype(float).iloc[0]
        return pd.Series({a: 0.0 for a in anos})


    # 2) Pegar contas (as "livres" ficam exatamente como o usuário digitou)
    receita = get("Receita Líquida")

    cmv = get("CMV, CPV ou CSP")  # sempre negativo
    lucro_bruto = receita + cmv

    desp_vendas = get("Despesas de Vendas")  # sempre negativo
    desp_ga = get("Despesas gerais e administrativas")  # sempre negativo
    outras_oper = get("Outras despesas/receitas operacionais")  # LIVRE (respeita sinal)
    da = get("Depreciação & Amortização")  # sempre negativo

    ebit = lucro_bruto + desp_vendas + desp_ga + outras_oper + da

    fin = get("Resultado Financeiro")  # LIVRE
    outros_nonop = get("Outros Resultados Não Operacionais")  # LIVRE
    lair = ebit + fin + outros_nonop

    imposto = get("Imposto de Renda")  # sempre negativo
    lucro_liq = lair + imposto

    ebitda = ebit - da  # add-back: como DA é negativo, subtrair DA soma

    # 3) Escrever no df (auto)
    def set_row(nome, serie):
        mask = df_idx.index.map(conta_limpa) == nome
        if mask.any():
            for a in anos:
                df_idx.loc[mask, a] = float(serie[a])


    set_row("Lucro Bruto", lucro_bruto)
    set_row("Lucro Operacional - EBIT", ebit)
    set_row("Lucro Antes do IR", lair)
    set_row("Lucro Líquido", lucro_liq)
    set_row("EBITDA", ebitda)

    # 4) Override (se preenchido substitui o auto)
    for total in ["Lucro Bruto", "Lucro Operacional - EBIT", "Lucro Antes do IR", "Lucro Líquido", "EBITDA"]:
        if total in df_idx.index and total in override_df.index:
            for a in anos:
                ovr = override_df.loc[total, a]
                if pd.notna(ovr):
                    df_idx.loc[total, a] = float(ovr)

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
        # -----------------------------
        # DRE
        # -----------------------------
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

        # Base inicial (se já houver em session_state, reaproveita)
        if "dre_raw" not in st.session_state:
            dre_base = pd.DataFrame({"Conta": dre_contas})
            for a in anos:
                dre_base[a] = 0.0
            st.session_state["dre_raw"] = dre_base.copy()

        dre_raw = st.data_editor(
            st.session_state["dre_raw"],
            use_container_width=True,
            num_rows="fixed",
            key="dre_editor"
        )
        st.session_state["dre_raw"] = dre_raw.copy()

        # Override separado (somente contas consolidadoras)
        contas_consolidadoras_dre = [
            "Lucro Bruto",
            "Lucro Operacional - EBIT",
            "Lucro Antes do IR",
            "Lucro Líquido",
            "EBITDA",
        ]

        st.markdown("#### Override de Totais (DRE) — opcional")
        st.caption("Se preencher, o valor digitado substitui o cálculo automático. Se deixar vazio, o sistema calcula sozinho.")

        if "dre_override" not in st.session_state:
            st.session_state["dre_override"] = criar_override_df(contas_consolidadoras_dre, anos)

        dre_override = st.data_editor(
            st.session_state["dre_override"].reset_index(),
            use_container_width=True,
            num_rows="fixed",
            key="dre_override_editor"
        )
        dre_override = dre_override.set_index("Conta")
        st.session_state["dre_override"] = dre_override.copy()

        # Consolida (auto + override)
        dre_calc = consolidar_dre_com_override(dre_raw, dre_override)
        st.session_state["dre_df"] = dre_calc.copy()

        st.divider()

        # -----------------------------
        # BALANÇO
        # -----------------------------
        st.subheader("Balanço Patrimonial")

        balanco_contas = [
            "Caixa e Similares",
            "Contas a Receber",
            "Estoques",
            "Adiantamentos",
            "Outros ativos circulantes",
            "Ativo Circulante",

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

            "Empréstimos e Financiamentos (LP)",
            "Impostos (LP)",
            "Outras Contas a Pagar",
            "Passivo Não Circulante",

            "Capital Social",
            "Reserva de Lucros",
            "Resultados Acumulados",
            "Patrimônio Líquido",
        ]

        if "bp_raw" not in st.session_state:
            bp_base = pd.DataFrame({"Conta": balanco_contas})
            for a in anos:
                bp_base[a] = 0.0
            st.session_state["bp_raw"] = bp_base.copy()

        bp_raw = st.data_editor(
            st.session_state["bp_raw"],
            use_container_width=True,
            num_rows="fixed",
            key="balanco_editor"
        )
        st.session_state["bp_raw"] = bp_raw.copy()

        # Override separado (somente contas consolidadoras)
        contas_consolidadoras_bp = [
            "Ativo Circulante",
            "Ativo Não Circulante",
            "Passivo Circulante",
            "Passivo Não Circulante",
            "Patrimônio Líquido",
        ]

        st.markdown("#### Override de Totais (Balanço) — opcional")
        st.caption("Se preencher, o valor digitado substitui o cálculo automático. Se deixar vazio, o sistema calcula sozinho.")

        if "bp_override" not in st.session_state:
            st.session_state["bp_override"] = criar_override_df(contas_consolidadoras_bp, anos)

        bp_override = st.data_editor(
            st.session_state["bp_override"].reset_index(),
            use_container_width=True,
            num_rows="fixed",
            key="bp_override_editor"
        )
        bp_override = bp_override.set_index("Conta")
        st.session_state["bp_override"] = bp_override.copy()

        # Consolida (auto + override)
        bp_calc = consolidar_bp_com_override(bp_raw, bp_override)
        if bp_calc is None:
            st.session_state["balanco_df"] = bp_raw.copy()
            st.warning("Consolidação do BP retornou vazio; exibindo os dados brutos.")
        else:
            st.session_state["balanco_df"] = bp_calc.copy()


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
                    "LL (DRE)": ll,
                    "D&A (add-back)": da_addback,
                    "Δ WC (consumo +)": delta_wc,
                    "CFO": cfo,
                    "CFI (proxy ANC)": cfi,
                    "CFF (proxy Dívida+PL)": cff,
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
                "- **CFO**: lucro líquido ajustado por D&A e variação do capital de giro (ΔWC).\n"
                "- **CFI**: proxy por variação do Ativo Não Circulante (pode diferir de CAPEX real).\n"
                "- **CFF**: proxy por variação de dívida e PL (não separa dividendos/juros pagos).\n"
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

            k1, k2, k3, k4, k5 = st.columns(5)
            k1.metric("PMR (dias)", f"{pmr_u:.0f}" if pd.notna(pmr_u) else "n/a")
            k2.metric("PME (dias)", f"{pme_u:.0f}" if pd.notna(pme_u) else "n/a")
            k3.metric("PMP (dias)", f"{pmp_u:.0f}" if pd.notna(pmp_u) else "n/a")
            k4.metric("Ciclo Operacional", f"{cop_u:.0f}" if pd.notna(cop_u) else "n/a")
            k5.metric("Ciclo Financeiro", f"{cfi_u:.0f}" if pd.notna(cfi_u) else "n/a")

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

                # Cards do último ano
                ultimo = anos_plot[-1]
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("IOG", f"R$ {float(iog[ultimo]):,.0f}".replace(",", "X").replace(".", ",").replace("X", "."))
                c2.metric("CPL", f"R$ {float(cpl[ultimo]):,.0f}".replace(",", "X").replace(".", ",").replace("X", "."))
                c3.metric("Saldo Tesouraria", f"R$ {float(st_saldo[ultimo]):,.0f}".replace(",", "X").replace(".", ",").replace("X", "."))
                c4.metric("Vendas", f"R$ {float(vendas[ultimo]):,.0f}".replace(",", "X").replace(".", ",").replace("X", "."))
