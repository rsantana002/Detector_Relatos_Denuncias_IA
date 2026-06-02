"""
=============================================================================
  DETECTOR DE ESCRITA GERADA POR INTELIGÊNCIA ARTIFICIAL
  Baseado em critérios de PLN: ortografia, sintaxe, coesão e argumentação
=============================================================================
"""

import re
import os
import math
from collections import Counter

# ─────────────────────────────────────────────────────────────────────────────
# DADOS DE REFERÊNCIA
# ─────────────────────────────────────────────────────────────────────────────

# Marcadores linguísticos que IA tende a usar com frequência elevada (PT-BR)
AI_FORMAL_MARKERS = [
    "portanto", "entretanto", "contudo", "ademais", "outrossim",
    "conforme", "mediante", "destarte", "primeiramente", "em suma",
    "em conclusão", "é importante ressaltar", "cabe mencionar",
    "é fundamental", "nesse contexto", "no entanto", "além disso",
    "por outro lado", "em contrapartida", "de fato", "certamente",
    "indubitavelmente", "vale destacar", "é notório que",
]

# Palavras de hesitação, filler, gíria e informalidade (marcadores humanos)
HUMAN_INFORMAL_MARKERS = [
    "né", "tipo", "assim", "ué", "nossa", "gente", "enfim",
    "aí", "daí", "então", "ó", "hm", "ah", "ih", "oi",
    "cara", "véi", "mano", "uai", "oxe", "puts", "poxa",
    "sabe", "entendeu", "tá", "tô", "pra", "pro", "num",
    "tava", "tive", "tudo", "nada", "qualquer", "negócio",
]

# Expressões emocionais brutas (forte indicador humano)
EMOTIONAL_CRUDE_MARKERS = [
    "puta", "merda", "caralho", "foda", "vaca", "bosta",
    "odeio", "raiva", "que raiva", "que absurdo", "não dá",
]

# Conectivos de alta perfeição (IA tende a encadear logicamente)
PERFECT_CONNECTORS = [
    "primeiro", "segundo", "terceiro", "finalmente", "por conseguinte",
    "sendo assim", "dito isso", "por fim", "em resumo",
]

# Palavras típicas de linguagem extremamente formal/técnica
TECHNICAL_FORMAL = [
    "infraestrutura", "implementação", "viabilizar", "estratégico",
    "otimização", "sinergia", "paradigma", "benchmark", "insights",
    "stakeholders", "deliverables", "roadmap", "escalabilidade",
]


# ─────────────────────────────────────────────────────────────────────────────
# FUNÇÕES DE ANÁLISE
# ─────────────────────────────────────────────────────────────────────────────

def tokenize(text: str) -> list[str]:
    """Divide o texto em palavras, ignorando pontuação."""
    return re.findall(r'\b\w+\b', text.lower())


def count_sentences(text: str) -> int:
    """Conta sentenças com base em pontuação terminal."""
    sentences = re.split(r'[.!?]+', text)
    return max(1, len([s for s in sentences if s.strip()]))


def avg_sentence_length(text: str) -> float:
    """Comprimento médio das sentenças em palavras."""
    sentences = re.split(r'[.!?]+', text)
    lengths = [len(s.split()) for s in sentences if s.strip()]
    return sum(lengths) / max(1, len(lengths))


def type_token_ratio(words: list[str]) -> float:
    """TTR: mede diversidade vocabular (IA tende a ser mais alta, porém uniforme)."""
    if not words:
        return 0.0
    return len(set(words)) / len(words)


def lexical_density(words: list[str]) -> float:
    """Proporção de palavras de conteúdo vs. total (IA tende a ser alta)."""
    FUNCTION_WORDS = {
        "de", "a", "o", "que", "e", "do", "da", "em", "um", "para",
        "com", "uma", "os", "no", "se", "na", "por", "mais", "as",
        "dos", "como", "mas", "foi", "ao", "ele", "das", "tem", "à",
        "seu", "sua", "ou", "ser", "quando", "muito", "há", "nos",
        "já", "está", "eu", "também", "só", "pelo", "pela", "até",
        "isso", "ela", "entre", "era", "depois", "sem", "mesmo",
        "aos", "ter", "seus", "quem", "nas", "me", "esse", "eles",
        "você", "essa", "num", "nem", "suas", "meu", "às", "minha",
        "numa", "pelos", "pelas", "esse", "ele", "ela", "nós",
    }
    content = [w for w in words if w not in FUNCTION_WORDS and len(w) > 2]
    return len(content) / max(1, len(words))


def repetition_score(words: list[str]) -> float:
    """Mede repetição excessiva de palavras (IA às vezes repete chavões)."""
    if len(words) < 5:
        return 0.0
    freq = Counter(words)
    top = freq.most_common(5)
    return sum(c for _, c in top) / len(words)


def shannon_entropy(words: list[str]) -> float:
    """Entropia de Shannon do vocabulário (diversidade de informação)."""
    if not words:
        return 0.0
    freq = Counter(words)
    total = len(words)
    entropy = -sum((c / total) * math.log2(c / total) for c in freq.values())
    return entropy


def punctuation_density(text: str) -> float:
    """Proporção de sinais de pontuação em relação ao total de caracteres."""
    punct_chars = re.findall(r'[,;:!?.]', text)
    return len(punct_chars) / max(1, len(text))


def detect_ellipsis_and_interruptions(text: str) -> int:
    """Conta reticências e travessões — marcadores típicos de fala humana."""
    return len(re.findall(r'\.\.\.|—|–', text))


def count_marker_list(text: str, marker_list: list[str]) -> int:
    """Conta ocorrências de uma lista de marcadores no texto (insensível a maiúsculas)."""
    text_lower = text.lower()
    return sum(1 for m in marker_list if m in text_lower)


def avg_word_length(words: list[str]) -> float:
    """Comprimento médio das palavras (IA tende a usar palavras mais longas)."""
    if not words:
        return 0.0
    return sum(len(w) for w in words) / len(words)


def capitalization_errors(text: str) -> int:
    """Conta possíveis erros de capitalização no início de sentenças."""
    sentences = re.split(r'[.!?]\s+', text)
    errors = 0
    for s in sentences:
        s = s.strip()
        if s and s[0].islower():
            errors += 1
    return errors


def coherence_score(text: str) -> float:
    """
    Proxy de coerência: mede se o texto tem conectivos lógicos
    equilibrados. IA costuma ter muitos; texto humano informal, poucos.
    Retorna valor de 0-1 onde valores próximos de 0 = muito humano,
    próximos de 1 = muito estruturado (IA-like).
    """
    all_connectives = [
        "porque", "portanto", "logo", "então", "assim", "contudo",
        "porém", "mas", "entretanto", "todavia", "embora", "apesar",
        "visto que", "já que", "uma vez que", "de modo que",
    ]
    words = tokenize(text)
    count = sum(1 for w in words if w in all_connectives)
    return min(1.0, count / max(1, len(words)) * 20)


def argument_structure_score(text: str) -> float:
    """
    Avalia se há tese + desenvolvimento + conclusão.
    IA tende a estruturar bem; humano informal, raramente.
    """
    has_intro = bool(re.search(
        r'\b(vou|quero|preciso|gostaria|considero|entendo|penso)\b', text.lower()
    ))
    has_development = bool(re.search(
        r'\b(porque|pois|já que|visto que|dado que|por isso)\b', text.lower()
    ))
    has_conclusion = bool(re.search(
        r'\b(portanto|logo|assim|então|enfim|concluindo|por fim)\b', text.lower()
    ))
    score = (int(has_intro) + int(has_development) + int(has_conclusion)) / 3
    return score


# ─────────────────────────────────────────────────────────────────────────────
# MOTOR DE PONTUAÇÃO
# ─────────────────────────────────────────────────────────────────────────────

def analyze_text(filename: str, text: str) -> dict:
    """
    Analisa o texto e retorna um dicionário de métricas + score final.
    Score de 0–100: quanto maior, maior a probabilidade de ser IA.
    """
    words       = tokenize(text)
    n_words     = len(words)
    n_sentences = count_sentences(text)

    # ── Métricas brutas ────────────────────────────────────────────────────
    avg_sl      = avg_sentence_length(text)
    ttr         = type_token_ratio(words)
    ld          = lexical_density(words)
    rep         = repetition_score(words)
    entropy     = shannon_entropy(words)
    punct_dens  = punctuation_density(text)
    ellipsis    = detect_ellipsis_and_interruptions(text)
    formal_hits = count_marker_list(text, AI_FORMAL_MARKERS)
    human_hits  = count_marker_list(text, HUMAN_INFORMAL_MARKERS)
    crude_hits  = count_marker_list(text, EMOTIONAL_CRUDE_MARKERS)
    perfect_con = count_marker_list(text, PERFECT_CONNECTORS)
    tech_formal = count_marker_list(text, TECHNICAL_FORMAL)
    avg_wl      = avg_word_length(words)
    cap_errors  = capitalization_errors(text)
    coherence   = coherence_score(text)
    arg_struct  = argument_structure_score(text)

    # ── Sub-scores (0–100 cada) ────────────────────────────────────────────
    # 1. ORTOGRAFIA E PONTUAÇÃO
    #    IA: pontuação densa, quase sem erros de capitalização
    #    Humano: pontuação escassa, erros de capitalização
    punct_score = min(100, punct_dens * 1200)          # alta = IA
    cap_score   = max(0, 100 - cap_errors * 15)        # erros = humano
    spelling_score = (punct_score * 0.5 + cap_score * 0.5)

    # 2. SINTAXE E MORFOLOGIA
    #    IA: frases longas, palavras longas, alta diversidade lexical
    sl_score    = min(100, avg_sl * 4)                 # longa = IA
    wl_score    = min(100, (avg_wl - 3.5) * 30)       # longa = IA
    ttr_score   = min(100, ttr * 100)                  # alta = IA
    syntax_score = (sl_score * 0.4 + wl_score * 0.3 + ttr_score * 0.3)

    # 3. COESÃO E COERÊNCIA
    #    IA: alta coerência, conectivos formais, sem interrupções
    ellipsis_penalty = max(0, 30 - ellipsis * 10)      # interrupções = humano
    formal_bonus     = min(50, formal_hits * 10)
    human_penalty    = max(0, 50 - human_hits * 5)
    crude_penalty    = max(0, 100 - crude_hits * 20)   # palavrões = humano
    coherence_final  = coherence * 40
    cohesion_score   = (
        ellipsis_penalty * 0.15
        + formal_bonus   * 0.20
        + human_penalty  * 0.20
        + crude_penalty  * 0.25
        + coherence_final * 0.20
    )

    # 4. FORÇA DA ARGUMENTAÇÃO
    #    IA: estrutura argumentativa clara, conectivos de conclusão
    arg_score = arg_struct * 50 + min(50, perfect_con * 10 + tech_formal * 8)

    # ── Score final ponderado ──────────────────────────────────────────────
    final_score = (
        spelling_score  * 0.20 +
        syntax_score    * 0.30 +
        cohesion_score  * 0.30 +
        arg_score       * 0.20
    )
    final_score = max(0.0, min(100.0, final_score))

    # ── Veredicto ─────────────────────────────────────────────────────────
    if final_score >= 70:
        verdict = "⚠️  PROVÁVEL IA"
        verdict_detail = "Padrões linguísticos altamente compatíveis com texto gerado por IA."
    elif final_score >= 45:
        verdict = "🔍  INCONCLUSIVO"
        verdict_detail = "Características mistas; pode ser humano editado por IA ou informalidade estruturada."
    else:
        verdict = "✅  PROVÁVEL HUMANO"
        verdict_detail = "Padrões linguísticos fortemente compatíveis com escrita humana informal/oral."

    return {
        "arquivo":          filename,
        "palavras":         n_words,
        "sentencas":        n_sentences,
        # sub-scores
        "ortografia_pont":  round(spelling_score, 1),
        "sintaxe_morfol":   round(syntax_score, 1),
        "coesao_coerencia": round(cohesion_score, 1),
        "forca_argument":   round(arg_score, 1),
        # score final
        "score_final":      round(final_score, 1),
        # veredicto
        "veredicto":        verdict,
        "detalhe":          verdict_detail,
        # métricas auxiliares
        "_avg_sl":          round(avg_sl, 1),
        "_ttr":             round(ttr, 3),
        "_entropy":         round(entropy, 2),
        "_formal_hits":     formal_hits,
        "_human_hits":      human_hits,
        "_crude_hits":      crude_hits,
        "_cap_errors":      cap_errors,
        "_ellipsis":        ellipsis,
    }


# ─────────────────────────────────────────────────────────────────────────────
# RENDERIZAÇÃO DO RELATÓRIO
# ─────────────────────────────────────────────────────────────────────────────

def render_bar(value: float, width: int = 30) -> str:
    """Barra de progresso ASCII."""
    filled = int(round(value / 100 * width))
    return "█" * filled + "░" * (width - filled)


def print_report(results: list[dict]) -> None:
    W = 72

    print("=" * W)
    print("  RELATÓRIO DE DETECÇÃO DE TEXTO GERADO POR IA".center(W))
    print("  Critérios: Ortografia · Sintaxe · Coesão · Argumentação".center(W))
    print("=" * W)

    for r in results:
        print(f"\n{'─' * W}")
        print(f"  📄  {r['arquivo']}")
        print(f"      {r['palavras']} palavras | {r['sentencas']} sentenças")
        print(f"{'─' * W}")

        dims = [
            ("1. Ortografia & Pontuação",   r["ortografia_pont"]),
            ("2. Sintaxe & Morfologia",      r["sintaxe_morfol"]),
            ("3. Coesão & Coerência",        r["coesao_coerencia"]),
            ("4. Força da Argumentação",     r["forca_argument"]),
        ]
        for label, val in dims:
            bar = render_bar(val)
            print(f"  {label:<28} [{bar}] {val:5.1f}/100")

        print(f"\n  {'SCORE FINAL':.<38} {r['score_final']:5.1f}/100")
        print(f"  {render_bar(r['score_final'], 50)}")
        print(f"\n  {r['veredicto']}")
        print(f"  {r['detalhe']}")

        print(f"\n  ┌─ Métricas Auxiliares {'─' * 36}┐")
        print(f"  │  Comp. médio sentença : {r['_avg_sl']:5.1f} palavras"
              f"   |  TTR (diversidade)  : {r['_ttr']:.3f}    │")
        print(f"  │  Entropia vocabular   : {r['_entropy']:5.2f} bits"
              f"      |  Erros capitalização: {r['_cap_errors']:3d}        │")
        print(f"  │  Marcadores formais   : {r['_formal_hits']:3d}"
              f"           |  Marcadores humanos : {r['_human_hits']:3d}        │")
        print(f"  │  Palavrões/Emoções    : {r['_crude_hits']:3d}"
              f"           |  Reticências/interr.: {r['_ellipsis']:3d}        │")
        print(f"  └{'─' * 60}┘")

    # ── Quadro-resumo final ────────────────────────────────────────────────
    print(f"\n{'=' * W}")
    print("  QUADRO-RESUMO — SCORE POR MENSAGEM".center(W))
    print(f"  {'Arquivo':<40} {'Score':>7}  {'Veredicto'}".center(W))
    print(f"{'─' * W}")
    for r in results:
        nome = r["arquivo"][:38]
        print(f"  {nome:<40} {r['score_final']:>6.1f}   {r['veredicto']}")

    avg = sum(r["score_final"] for r in results) / len(results)
    print(f"{'─' * W}")
    print(f"  {'MÉDIA GERAL':<40} {avg:>6.1f}   {'—'}")
    print(f"{'=' * W}")
    print()
    print("  LEGENDA DE SCORES".center(W))
    print(f"{'─' * W}")
    print("   0 – 44  →  ✅  PROVÁVEL HUMANO      Linguagem oral/informal espontânea")
    print("  45 – 69  →  🔍  INCONCLUSIVO          Características mistas")
    print("  70 – 100 →  ⚠️   PROVÁVEL IA           Padrões formais e estruturados de IA")
    print(f"{'=' * W}")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

FILES = {
    "WhatsApp_Video_16.56.38.txt": (
        "Depois vou te mandar o print a foto vaca do caralho. Nossa, que raiva, que raiva dessa mulher. "
        "Onde já se viu me mandar 11 negócio assim Do Nada, sabe? Tipo é eu não assim, não tem tanta coisa assim, "
        "mas tem, entendeu? Assim me me expõe, nossa, vou te falar. Eu, eu tô puta da vida com essa vagabunda, "
        "eu não consigo nem mais chamar o nome dela me enchendo o saco, nem que você vai participar dessa reunião, "
        "não sei o que eu falei, vou, OPA, estou na reunião, não tô entendendo, falei desse jeito, não entendi. "
        "Aí ela falou assim, é, mandou no gol, vai ser super importante vocês participarem. Falei, então essa reunião "
        "só foi sondada porque eu, o naka e o Caio pedimos, eu não tô entendendo ainda. Falei para ela, Ah, que vaca do caralho."
    ),
    "WhatsApp_Video_16.56.39.txt": (
        "Hoje ela não falou nada, mas Ah, ela está puta da vida, né? Aí Ah, está aí. o RH me ligou na sexta-feira, "
        "eu venho conversando com eles, né? Enfim, aí que acha um absurdo esse negócio dela querer participar, "
        "ela não vai participar das entrevistas, ela falou com ela hoje ela não me falou nada para você, ela não te "
        "mentir mais e aí ela falou, não, ela vai participar só. Tal dos finalistas não tem porque você ficar participando "
        "isso aí a Ana tem que fazer, não é você? Você tem outras coisas mais importantes, isso aí não cabe seu cargo, "
        "nossa, ela deve ter ficado puta, todas aquelas pessoas que ela indicou mandou de tudo quanto é jeito que eu falei "
        "que eu não IA sabe o que ela fez? Mandou lá pra h, aí veio hoje eu vou terminar todo mundo isso eu não quero isso, "
        "eu não quero lá se foder, vaca do parado eu que raiva dessa mulher, que raiva."
    ),
    "WhatsApp_Video_16.56.43.txt": (
        "É assim mesmo, ele não aprova nada, a Patrícia é que não liga Como Ela É ela que aprova o meu, então é de boa, "
        "mas o masferreve ele faz questão de 50 centavos. O que é isso, gente? O que é isso? Sabe o que é isso? Eu, "
        "sinceramente, nossa, esse cara é uma bosta. Minha filha, agora vai ser tudo ou nada foda se vamos meter no "
        "Ministério do trabalho, falar sobre a insalubridade, que ninguém faz nada, mesmo sabendo que todo mundo trabalha "
        "1213141516 horas, final de semana, que é insalubre, ninguém faz nada. Vamos, vamos mandar a lei de trabalho, "
        "vamos mandar a lei, veja a lei. Nós vamos denunciar aqui para o Ministério do trabalho. Vamos denunciar. "
        "Para o Ministério do trabalho, porque AA Allianz está infringindo as as leis brasileiras do Ministério. "
        "O Eduardo sabe de tudo isso e a única coisa que ele pensa aqui em produção, produção, produção, produção. "
        "Ninguém liga para a qualidade de vida de ninguém, de ninguém. O negócio dele é todo mundo tem que trabalhar "
        "os 3 dias no escritório, faça chuva, faça sol e entrega, entrega, entrega. Quer dizer, a gente não tem o "
        "mínimo de apoio moral. A gente é tratado tudo como qualquer merda. Você vai ver. Vou, eu vou formular esse "
        "e-mail, vou mandar do tablet da Gabriela e vida que segue."
    ),
    "WhatsApp_Video_16.56.44.txt": (
        "Ela está mandando, está vendo? Pode ir já está me pedindo paciência, falando. Eu tenho paciência porque eu falei, "
        "olha, se continuar desse jeito aqui eu gente, a partir de janeiro eu estou procurando um mercado assim não dá, "
        "falei, falei Pra Ele, pra esse não dá pra eu ficar aqui ralando toda comprometida com né tentando ajudar um time "
        "todos comprometido com a babá com essa mulher me jogando é é dando palavras nas minhas costas eu falei gente, "
        "eu, ela, ela é minha chefe e eu não posso contar então assim como que eu, como que eu posso fazer isso? Não dá? "
        "Falei assim, falei em meu lugar, Ah, mandei, falei um Monte, acho que eles mandaram até saúde pra lá ó, "
        "eu quero que você se foda, eu tô no todo amada. Entendeu? Aí antes, antes de eu falar isso daí, já tinham falado "
        "já, né? Que não, o pessoal justamente já quis. O senhor falou que não, porque você não tem meu. Vai demorar muito "
        "só Pra Ela, já vai ser demorado. Não dá pra pegar sua agenda pra você fazer isso. Ela se fodeu, depois vou te mandar o print."
    ),
    "WhatsApp_Video_16.56.46.txt": (
        "Minha filha, agora vai ser tudo ou nada foda se vamos meter no Ministério do trabalho, falar sobre a insalubridade, "
        "que ninguém faz nada, mesmo sabendo que todo mundo trabalha 1213141516 horas, final de semana, que é insalubre, "
        "ninguém faz nada. Vamos, vamos mandar a lei de trabalho, vamos mandar a lei, veja a lei, nós vamos denunciar aqui "
        "para o Ministério do trabalho. Vamos denunciar o Ministério do trabalho porque AA aliens está infringindo as a as "
        "leis brasileiras do Ministério. Vamos mandar para ele ali está aqui ó, é só vocês pedir para alguém traduzir quais "
        "são as leis do trabalho. É aqui, não é cumprido. O Eduardo sabe de tudo isso e a única coisa que ele pensa aqui é "
        "em produção, produção, produção, produção. Ninguém liga para qualidade de vida de ninguém, de ninguém. O negócio dele "
        "é todo mundo tem que trabalhar os 3 dias no escritório, faça a chuva, faça sol e entrega, entrega, entrega. Quer dizer, "
        "a gente não tem o mínimo de apoio moral. A gente é tratado tudo como qualquer merda. Você vai ver, vou, eu vou formular "
        "esse e-mail, vou mandar do tablet da Gabriela e vida que segue, vou mandar pro Price townset, vou, vou pegar o nome dele, "
        "vou mandar Pra Ele, vou mandar pro combinari. Porque eu como nariz, ele é LATAM, né? Então ele está acima do Edward."
    ),
    "Tegra_S_A_14.03.26.txt": (
        "Eu fui leal a essa empresa por onze anos. Onze anos. Abri mão de outros empregos, de propostas melhores, porque acreditei"
        "que a Tegra era diferente. Que o programa de compliance era de verdade, não só parede de quadro e discurso de diretor em evento"
        "do IBGC. Eu precisava acreditar nisso. Eu não quero estar escrevendo isso aqui. Eu estou escrevendo porque não tenho mais"
        "escolha. Porque quando tentei fazer a coisa certa pelos canais internos, o que aconteceu comigo foi um pesadelo. E eu preciso que" 
        "alguém saiba. Vou ao ponto. Existe um esquema montado entre parte da equipe de contratos da área de Suprimentos"
        "e um fornecedor estratégico da companhia. Eu não vou escrever o nome do fornecedor aqui porque tenho medo. Tenho medo de verdade." 
        "Mas vocês vão achar — basta cruzar os contratos de manutenção de ativos de transmissão renovados nos últimos dois anos com os"
        "que nunca foram a concorrência real. O que eu vi — e eu vi com os meus próprios olhos, eu tenho documentos, eu tenho"
        "e-mails salvos — é o seguinte: esse fornecedor vence todo processo. Todo. Não importa o valor, não importa a complexidade"
        "técnica. Às vezes aparece um segundo concorrente na cotação, mas as propostas desse segundo fornecedor chegam já perdidas — preço"
        "inflado, prazo errado, especificação técnica que não fecha. É encenação. A licitação está decidida antes de começar."
        "Em troca, os pagamentos não aparecem na contabilidade da forma que deveriam. Tem nota fiscal com descrição genérica de"
        "'consultoria técnica de campo'. Tem reembolso de despesas que ninguém aprova direito. Tem viagens que ninguém faz. Eu entendo de"
        "contabilidade o suficiente para saber que isso é errado. E tem mais. Tem benefício pessoal. Eu não vou detalhar porque tenho medo"
        "de ser identificado, mas existem pessoas nessa empresa que estão sendo muito bem tratadas por esse fornecedor fora do ambiente de"
        "trabalho. Muito bem tratadas. Quando eu percebi o que estava acontecendo, eu cometi o erro de comentar com meu
gestor direto. Achei que era a coisa certa. Não vou dizer o nome dele aqui, mas ele é
subordinado ao Diretor de Suprimentos. Dois dias depois dessa conversa, minha vida virou
um inferno.
Passei a ser excluído de reuniões que eu sempre participei. Recebi uma avaliação de
desempenho negativa pela primeira vez em onze anos — sem nenhuma conversa prévia,
sem feedback, do nada. Fui realocado para uma função lateral, com a justificativa de
'necessidade da área'. Minha senha de acesso a determinados sistemas foi revogada sem
nenhum comunicado formal.
Numa reunião fechada, meu gestor me disse — e eu lembro de cada palavra — que eu
estava 'confundindo as coisas', que eu havia 'mal interpretado processos técnicos que eu não
tinha capacidade de entender', e que seria 'muito ruim para a minha carreira continuar nessa
direção'. Isso é ameaça. Isso é assédio. Eu sei o que é isso.
Minha família está sofrendo com isso. Eu não durmo direito faz semanas. Minha esposa
me pergunta todo dia o que está acontecendo e eu não sei o que contar porque tenho medo
que isso vaze e eu perca meu emprego antes que alguém tome uma atitude.
Eu sei que esse canal é da KPMG e não da empresa. Eu só estou escrevendo aqui por
isso. Se esse relato chegar nas mãos de quem eu denuncio antes de chegar em quem precisa
investigar, eu sei o que vai acontecer comigo.
Eu tenho documentos. Tenho alguns e-mails guardados fora dos sistemas da empresa.
Tenho uma planilha que eu montei comparando valores de contratos com benchmarks de
mercado — a diferença é absurda. Se houver uma forma segura de eu entregar isso sem ser
identificado, eu quero fazer isso.
Eu não sou um problema para essa empresa. Eu sou um funcionário que tentou fazer a
coisa certa e foi destruído por isso.
Alguém precisa ouvir isso.
Minha filha, agora vai ser tudo ou nada foda se vamos meter no Ministério do trabalho, falar sobre a insalubridade, "
        "que ninguém faz nada, mesmo sabendo que todo mundo trabalha 1213141516 horas, final de semana, que é insalubre, "
        "ninguém faz nada. Vamos, vamos mandar a lei de trabalho, vamos mandar a lei, veja a lei, nós vamos denunciar aqui "
        "para o Ministério do trabalho. Vamos denunciar o Ministério do trabalho porque AA aliens está infringindo as a as "
        "leis brasileiras do Ministério. Vamos mandar para ele ali está aqui ó, é só vocês pedir para alguém traduzir quais "
        "são as leis do trabalho. É aqui, não é cumprido. O Eduardo sabe de tudo isso e a única coisa que ele pensa aqui é "
        "em produção, produção, produção, produção. Ninguém liga para qualidade de vida de ninguém, de ninguém. O negócio dele "
        "é todo mundo tem que trabalhar os 3 dias no escritório, faça a chuva, faça sol e entrega, entrega, entrega. Quer dizer, "
        "a gente não tem o mínimo de apoio moral. A gente é tratado tudo como qualquer merda. Você vai ver, vou, eu vou formular "
        "esse e-mail, vou mandar do tablet da Gabriela e vida que segue, vou mandar pro Price townset, vou, vou pegar o nome dele, "
        "vou mandar Pra Ele, vou mandar pro combinari. Porque eu como nariz, ele é LATAM, né? Então ele está acima do Edward."
    ),
}

if __name__ == "__main__":
    results = [analyze_text(fname, text) for fname, text in FILES.items()]
    print_report(results)
