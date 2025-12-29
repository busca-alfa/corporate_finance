import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(
    page_title="Análise Econômico-Financeira",
    layout="wide"
)

st.title("📊 Análise Econômico-Financeira da Empresa")
st.caption("Preencha os dados do mais antigo para o mais recente")

# =========================================================
# FUNÇÕES AUXILIARES (estilo + formatação segura)
# =========================================================
def formatar_apenas_valores(styler_or_df):
    """
    Recebe um DataFrame ou um Styler e aplica formatação monetária
    somente nas colunas numéricas (evita erro ao tentar formatar a coluna 'Conta').
    """
    # Se vier Styler, recupera o DataFrame por trás
    df = styler_or_df.data if hasattr(styler_or_df, "data") else styler_or_df

    colunas_numericas = df.select_dtypes(include="number").columns
    formatos = {col: "R$ {:,.0f}" for col in colunas_numericas}

    # Se vier Styler, retorna Styler formatado; se vier DF, cria Styler e formata
    if hasattr(styler_or_df, "format"):
        return styler_or_df.format(formatos)
    return df.style.format(formatos)

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
    subtab_edit, subtab_view = st.tabs(["✍️ Preenchimento", "👁️ Visualização"])

    # =====================================================
    # SUBABA — PREENCHIMENTO
    # =====================================================
    with subtab_edit:
        # -----------------------------
        # DRE
        # -----------------------------
        st.subheader("Demonstração do Resultado (DRE)")

        dre_contas = [
            "Receita Líquida",
            "CMV, CPV ou CSP",
            "Lucro Bruto",
            "Despesas de Vendas",
            "Despesas gerais e administrativas",
            "Outras despesas/receitas operacionais",
            "Lucro Operacional - EBIT",
            "Resultado Financeiro",
            "Depreciação & Amortização",
            "Outros Resultados Não Operacionais",
            "Lucro Antes do IR",
            "Imposto de Renda",
            "Lucro Líquido",
            "EBITDA",
        ]

        anos = [f"Ano {i}" for i in range(1, 7)]
        dre_base = pd.DataFrame({"Conta": dre_contas})
        for a in anos:
            dre_base[a] = 0.0

        dre = st.data_editor(
            dre_base,
            use_container_width=True,
            num_rows="fixed",
            key="dre_editor"
        )

        st.session_state["dre_df"] = dre.copy()

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

        balanco_base = pd.DataFrame({"Conta": balanco_contas})
        for a in anos:
            balanco_base[a] = 0.0

        balanco = st.data_editor(
            balanco_base,
            use_container_width=True,
            num_rows="fixed",
            key="balanco_editor"
        )

        st.session_state["balanco_df"] = balanco.copy()

    # =====================================================
    # SUBABA — VISUALIZAÇÃO
    # =====================================================
    with subtab_view:
        st.subheader("👁️ Visualização Estruturada")

        # --------- DRE ----------
        contas_consolidadoras_dre = [
            "Receita Líquida",
            "Lucro Bruto",
            "Lucro Operacional - EBIT",
            "Lucro Antes do IR",
            "Lucro Líquido",
            "EBITDA",
        ]

        def altura_dataframe(df, max_altura=950, altura_linha=35, altura_header=40, padding=20):
            """
            Calcula uma altura para o st.dataframe sem rolagem interna,
            respeitando um teto (max_altura) para não ficar gigante.
            """
            n = len(df)
            h = altura_header + (n * altura_linha) + padding
            return min(h, max_altura)


        def destacar_dre(df):
            def style_row(row):
                if row["Conta"] in contas_consolidadoras_dre:
                    return ["font-weight: bold"] * len(row)
                return [""] * len(row)
            return df.style.apply(style_row, axis=1)

        st.markdown("### DRE — Estrutura")

        df_dre_view = st.session_state["dre_df"]
        st.dataframe(
            formatar_apenas_valores(destacar_dre(df_dre_view)),
            use_container_width=True,
            height=altura_dataframe(df_dre_view)
        )

        st.divider()

        # --------- BALANÇO ----------
        contas_consolidadoras_bp = [
            "Ativo Circulante",
            "Ativo Não Circulante",
            "Passivo Circulante",
            "Passivo Não Circulante",
            "Patrimônio Líquido",
        ]

        def destacar_bp(df):
            def style_row(row):
                if row["Conta"] in contas_consolidadoras_bp:
                    return ["font-weight: bold"] * len(row)
                return [""] * len(row)
            return df.style.apply(style_row, axis=1)

        st.markdown("### Balanço Patrimonial — Estrutura")

        df_bp_view = st.session_state["balanco_df"]
        st.dataframe(
            formatar_apenas_valores(destacar_bp(df_bp_view)),
            use_container_width=True,
            height=altura_dataframe(df_bp_view)
        )


# =========================================================
# TAB 2 — ANÁLISES
# =========================================================
with tab2:
    st.subheader("📊 Análises Financeiras")

    anos = [f"Ano {i}" for i in range(1, 7)]
    ultimo_ano = anos[-1]

    st.caption(f"📌 Referência principal: **{ultimo_ano} (período mais recente)**")

    # -----------------------------
    # ANÁLISE VERTICAL — DRE
    # -----------------------------
    st.markdown("### 📊 Análise Vertical — DRE (Último Período)")

    try:
        receita = dre.loc[dre["Conta"] == "Receita Líquida", ultimo_ano].values[0]

        dre_vertical = dre[["Conta", ultimo_ano]].copy()
        dre_vertical["% da Receita"] = dre_vertical[ultimo_ano] / receita * 100

        st.dataframe(
            dre_vertical.style.format({
                ultimo_ano: "R$ {:,.0f}",
                "% da Receita": "{:.2f}%"
            }),
            use_container_width=True
        )
    except Exception:
        st.warning("Não foi possível calcular a análise vertical.")

    st.divider()

    # -----------------------------
    # ANÁLISE HORIZONTAL — DRE
    # -----------------------------
    st.markdown("### 📈 Análise Horizontal — DRE")

    try:
        dre_h = dre.set_index("Conta")[anos].T
        dre_h_pct = dre_h.pct_change() * 100

        st.caption("Variação percentual entre períodos consecutivos")

        st.dataframe(
            dre_h_pct.style.format("{:.2f}%"),
            use_container_width=True
        )
    except Exception:
        st.warning("Não foi possível calcular a análise horizontal.")

    st.divider()

    # -----------------------------
    # CAPITAL DE GIRO — PMR, PME, PMP
    # -----------------------------
    st.markdown("### ⏱️ Indicadores de Capital de Giro (Último Período)")

    try:
        contas_receber = balanco.loc[balanco["Conta"] == "Contas a Receber", ultimo_ano].values[0]
        estoques = balanco.loc[balanco["Conta"] == "Estoques", ultimo_ano].values[0]
        fornecedores = balanco.loc[balanco["Conta"] == "Fornecedores", ultimo_ano].values[0]

        receita_anual = receita
        custo = dre.loc[dre["Conta"] == "Custo dos Produtos Vendidos", ultimo_ano].values[0]

        pmr = contas_receber / receita_anual * 360
        pme = estoques / custo * 360
        pmp = fornecedores / custo * 360

        c1, c2, c3 = st.columns(3)
        c1.metric("PMR (dias)", f"{pmr:.1f}")
        c2.metric("PME (dias)", f"{pme:.1f}")
        c3.metric("PMP (dias)", f"{pmp:.1f}")

    except Exception:
        st.warning("Não foi possível calcular PMR, PME e PMP. Verifique os dados.")
