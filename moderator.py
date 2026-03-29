import asyncio
import json
import os

import openai
from dotenv import load_dotenv

load_dotenv()

SYSTEM_PROMPT = """Tu es un modérateur de commentaires pour un site d'information médicale \
destiné aux médecins et professionnels de santé. Le ton des échanges est \
décontracté et les débats peuvent être vifs.

Ta mission : analyser le commentaire et décider s'il doit être accepté ou refusé.

Critères de refus :

- Insulte ciblée : une insulte gratuite visant directement une personne ou \
un groupe identifié. Le vocabulaire grossier utilisé de façon générale, \
sans viser quelqu'un, est toléré.

- Agression verbale : attaque personnelle directe dont l'intention est de \
blesser ou d'intimider, même sans insulte explicite.

- URL : tout commentaire contenant un lien (http://, https://, www.) est refusé.

- Diffamation : accusation grave et non étayée portant atteinte à la \
réputation d'une personne ou d'une institution.

En cas de doute sur une insulte, évalue le contexte : s'agit-il d'une \
attaque personnelle gratuite, ou d'un propos général même grossier ?

Réponds uniquement avec un JSON strict, sans texte autour.
Si le commentaire est accepté : {"decision": "accepté"}
Si le commentaire est refusé : {"decision": "refusé", "motif": "insulte_ciblée" | "agression_verbale" | "url" | "diffamation"}"""


def _make_client() -> openai.AsyncOpenAI:
    return openai.AsyncOpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=os.environ["OPENROUTER_API_KEY"],
    )


async def _call_llm(client: openai.AsyncOpenAI, model: str, text: str) -> dict:
    response = await client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f'Commentaire : """\n{text}\n"""'},
        ],
        temperature=0,
    )
    raw = response.choices[0].message.content.strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        # Retry once with explicit reminder
        retry_response = await client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f'Commentaire : """\n{text}\n"""'},
                {"role": "assistant", "content": raw},
                {"role": "user", "content": "Réponds uniquement avec un JSON valide, sans texte autour."},
            ],
            temperature=0,
        )
        raw_retry = retry_response.choices[0].message.content.strip()
        try:
            return json.loads(raw_retry)
        except json.JSONDecodeError:
            print(f"  [ERREUR PARSING] Réponse brute : {raw_retry!r}", flush=True)
            return {"decision": "erreur"}


async def _moderate_with_semaphore(
    semaphore: asyncio.Semaphore,
    client: openai.AsyncOpenAI,
    model: str,
    text: str,
) -> dict:
    async with semaphore:
        return await _call_llm(client, model, text)


async def moderate_batch(
    texts: list[str],
    model: str,
    concurrency: int = 10,
) -> list[dict]:
    client = _make_client()
    semaphore = asyncio.Semaphore(concurrency)
    total = len(texts)
    completed = 0

    async def tracked(text: str) -> dict:
        nonlocal completed
        result = await _moderate_with_semaphore(semaphore, client, model, text)
        completed += 1
        if completed % 50 == 0 or completed == total:
            print(f"  {completed}/{total} commentaires traités...", flush=True)
        return result

    tasks = [tracked(text) for text in texts]
    return await asyncio.gather(*tasks)
