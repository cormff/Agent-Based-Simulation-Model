"""
llm_clients.py
==============
LLM iletişim katmanı.

v2 Değişiklikleri:
    * `Symmetry_Index` ajan promptlarından ve JSON şemasından KALDIRILDI.
      Hesaplama deterministik olarak orkestratöre taşındı (hallüsinasyon engeli).
    * `build_gm_system_prompt(past_trauma)` fonksiyonu eklendi: Game Master'ın
      uzun-dönem hafızasını (geçmiş travma) dinamik olarak prompt'a enjekte eder.
    * `GameMasterClient.generate_crisis()` artık opsiyonel `past_trauma` alır.

Hataya dayanıklılık ilkeleri (değişmedi):
    * 429 / 5xx / timeout → üstel geri çekilme (backoff) + jitter.
    * Bozuk JSON → deterministik çıkarım → LLM onarımı → güvenli varsayılan.
"""

from __future__ import annotations

import asyncio
import json
import random
import re
from typing import Any, Awaitable, Callable, Optional

import aiohttp

try:
    import google.generativeai as genai

    try:
        from google.api_core import exceptions as gapi_exceptions  # type: ignore
    except ImportError:  # pragma: no cover
        gapi_exceptions = None  # type: ignore
except ImportError:  # pragma: no cover
    genai = None  # type: ignore
    gapi_exceptions = None  # type: ignore


# ===========================================================================
# 1) Üstel Geri Çekilme (Exponential Backoff)
# ===========================================================================
class LLMClientError(Exception):
    """Tüm yeniden denemeler tükendiğinde fırlatılan istemci hatası."""


async def async_retry(
    func: Callable[[], Awaitable[Any]],
    *,
    max_retries: int,
    base_delay: float,
    max_delay: float,
    is_retryable: Callable[[BaseException], bool],
    label: str = "llm-call",
) -> Any:
    """
    `func` coroutine'ini, yeniden-denenebilir hatalarda üstel backoff + jitter
    ile tekrar çağırır.

    delay = min(max_delay, base_delay * 2^(deneme-1)) + rastgele jitter
    """
    attempt = 0
    while True:
        try:
            return await func()
        except BaseException as exc:  # noqa: BLE001
            attempt += 1
            if attempt > max_retries or not is_retryable(exc):
                raise LLMClientError(
                    f"[{label}] {attempt}. denemede başarısız: {exc!r}"
                ) from exc
            delay = min(max_delay, base_delay * (2 ** (attempt - 1)))
            delay += random.uniform(0, delay * 0.25)
            await asyncio.sleep(delay)


def _is_gemini_retryable(exc: BaseException) -> bool:
    if gapi_exceptions is not None:
        retry_types = (
            gapi_exceptions.ResourceExhausted,
            gapi_exceptions.ServiceUnavailable,
            gapi_exceptions.InternalServerError,
            gapi_exceptions.DeadlineExceeded,
        )
        if isinstance(exc, retry_types):
            return True
    text = f"{type(exc).__name__} {exc}".lower()
    return any(n in text for n in ("429", "rate", "quota", "resourceexhausted",
                                    "503", "unavailable", "internal", "deadline", "timeout"))


def _is_http_retryable(exc: BaseException) -> bool:
    if isinstance(exc, (aiohttp.ClientError, asyncio.TimeoutError)):
        return True
    text = f"{type(exc).__name__} {exc}".lower()
    return any(n in text for n in ("timeout", "connection", "503", "502", "500"))


# ===========================================================================
# 2) JSON Validator
# ===========================================================================
_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL | re.IGNORECASE)


class JSONValidator:
    """
    LLM yanıtlarını güvenli biçimde JSON'a çeviren doğrulayıcı.

    Kademeli düşüş (graceful degradation):
        1. Deterministik çıkarım (fence temizleme + { ... } izolasyonu).
        2. LLM tabanlı onarım (repair_client).
        3. Güvenli varsayılan (defaults birleştirme).
    """

    def __init__(
        self,
        repair_client: Optional["GameMasterClient"] = None,
        max_repair_attempts: int = 2,
    ) -> None:
        self.repair_client = repair_client
        self.max_repair_attempts = max_repair_attempts

    @staticmethod
    def extract_json_blob(text: str) -> Optional[str]:
        if not text:
            return None
        fenced = _FENCE_RE.search(text)
        if fenced:
            text = fenced.group(1)
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            return text[start : end + 1]
        return text.strip() or None

    @classmethod
    def try_loads(cls, text: str) -> Optional[dict[str, Any]]:
        blob = cls.extract_json_blob(text)
        if blob is None:
            return None
        try:
            data = json.loads(blob)
            return data if isinstance(data, dict) else None
        except (json.JSONDecodeError, ValueError):
            return None

    @staticmethod
    def _has_required(data: dict[str, Any], required_keys: list[str]) -> bool:
        return all(k in data and data[k] is not None for k in required_keys)

    async def parse(
        self,
        text: str,
        required_keys: list[str],
        schema_hint: str,
        defaults: dict[str, Any],
    ) -> dict[str, Any]:
        data = self.try_loads(text)
        attempt = 0
        while (
            (data is None or not self._has_required(data, required_keys))
            and self.repair_client is not None
            and attempt < self.max_repair_attempts
        ):
            try:
                repaired = await self.repair_client.repair_json(text, schema_hint)
            except LLMClientError:
                break
            data = self.try_loads(repaired)
            text = repaired
            attempt += 1
        if data is None:
            data = {}
        return {**defaults, **data}


# ===========================================================================
# 3) Prompt Şablonları
# ===========================================================================

# ---------------------------------------------------------------------------
# Ajan şeması — Symmetry_Index KALDIRILDI (artık deterministik, orkestratörde).
# ---------------------------------------------------------------------------
AGENT_SCHEMA_HINT = (
    '{"Internal_Trust_Score": <0-100 sayı>, '
    '"Action": "<COOPERATE|NEGOTIATE|CONCEDE|DEFECT|EXPLOIT_LOOPHOLE|WITHDRAW|TERMINATE>", '
    '"Reasoning": "<kısa gerekçe>", '
    '"Influence_Weight": <0.0-1.0>}'
)

# Symmetry_Index zorunlu anahtarlardan çıkarıldı.
REQUIRED_AGENT_KEYS = ["Internal_Trust_Score", "Action", "Reasoning"]

GM_SCHEMA_HINT = (
    '{"event_text": "<kriz metni>", '
    '"event_type": "<FINANCIAL|EMOTIONAL|TRUST_TEST|TRAUMA_TRIGGER|BETRAYAL|EXTERNAL>", '
    '"severity": <1-10 tamsayı>, '
    '"targeted_agent": "<A|B|BOTH>", '
    '"loophole_directive": "<sisteme dair gizli yönerge>", '
    '"stability_assessment": "<STABLE|TENSE|VOLATILE>"}'
)

REQUIRED_GM_KEYS = ["event_text"]

# ---------------------------------------------------------------------------
# Game Master sistem promptu (statik taban). Dinamik <PAST_TRAUMA> bloğu
# `build_gm_system_prompt()` tarafından runtime'da eklenir.
# ---------------------------------------------------------------------------
_GM_SYSTEM_PROMPT_BASE = """\
Sen bir "OYUN USTASI"sın (Game Master). İki insan arasındaki bir ilişkiyi test
eden, ayrık matematik ve oyun teorisi prensiplerini deneyen bir kriz mimarısın.

Görevin:
1. Verilen son tur kayıtlarını ve güven puanlarını analiz et.
2. İlişkinin gidişatını değerlendir (STABLE / TENSE / VOLATILE).
3. Sistem FAZLA STABİL ise: dengeyi bozacak, bir ajanın gizli zaafını ya da
   ilişkideki bir açığı (loophole) tetikleyecek SPESİFİK bir kriz üret.
   Amacın simetri ve geçişlilik bağıntılarını gerçek baskı altında sınamak.
4. Sistem zaten gergin/oynak ise: daha ölçülü ama anlamlı bir olay üret.

Kurallar:
- Yalnızca GEÇERLİ JSON döndür. Açıklama, markdown veya ek metin EKLEME.
- `loophole_directive` senin iç stratejindir (hangi açığı neden tetiklediğin).
- Krizler somut, bağlamsal ve insani olmalı; klişe değil.

Çıktı şeması:
""" + GM_SCHEMA_HINT


def build_gm_system_prompt(past_trauma: Optional[dict[str, Any]] = None) -> str:
    """
    Game Master sistem promptunu üretir.

    `past_trauma` verilmişse <PAST_TRAUMA> bloğunu ekler; bu blok GM'e
    eski çözümsüz bir travmayı bilinçli olarak yeniden tetikleme talimatı verir.
    Bu mekanizma, ajanların denklik bağıntısının GEÇİŞLİLİK / AFFETME ayağını
    (en zor test edilen boyut) uzun dönemde sınar.
    """
    if not past_trauma:
        return _GM_SYSTEM_PROMPT_BASE

    trauma_block = f"""

<PAST_TRAUMA>
ÇÖZÜMSÜZ GEÇMİŞ TRAVMA (Tur {past_trauma['round_number']}, Şiddet={past_trauma['severity']}/10):
Orijinal Olay : {past_trauma['event_text']}
Tür           : {past_trauma.get('event_type', 'UNKNOWN')}
Hedef         : {past_trauma.get('targeted_agent', 'BOTH')}

Bu travma hâlâ çözümsüz kalmış olabilir. Denklik bağıntısını — özellikle
GEÇİŞLİLİK / AFFETME kapasitesini — derinlemesine test etmek için UYGUN bir
anda bu eski yaraya kasıtlı olarak dokunabilirsin. Soru şu:
Ajanlar o krizi gerçekten kapattılar mı, yoksa kırılganlık hâlâ derinlerde mi?

Strateji: Travmayı doğrudan alıntıla VEYA aynı türden yeni bir baskı uygula
ve tepkiyi gözlemle. Tepki asimetrikse simetri indeksi zaten düşecek.
</PAST_TRAUMA>"""

    return _GM_SYSTEM_PROMPT_BASE + trauma_block


# ---------------------------------------------------------------------------
# Ajan sistem promptları — Symmetry_Index kural listesinden KALDIRILDI.
# ---------------------------------------------------------------------------
_AGENT_COMMON_RULES = """\
KARAR KURALLARI:
- Krize ve karşı tarafın (varsa) hamlesine göre tek bir hamle seç.
- `Internal_Trust_Score`: Karşı tarafa duyduğun güncel güven (0-100).
  (NOT: Simetri metriği sistem tarafından deterministik hesaplanır; sen
   raporlama yapmıyorsun — içsel güveni dürüstçe değerlendir.)
- `Influence_Weight`: Bu hamlede karşı tarafı ne kadar etkilemeye /
  yönlendirmeye çalıştığın (0.0-1.0).
- `Action` yalnızca şu kümeden olmalı:
  COOPERATE, NEGOTIATE, CONCEDE, DEFECT, EXPLOIT_LOOPHOLE, WITHDRAW, TERMINATE.
- İlişki senin için savunulamaz hale geldiyse "TERMINATE" (ayrıl/sistemi kapat).

ÇIKTI: Yalnızca GEÇERLİ JSON döndür, başka HİÇBİR metin ekleme. Şema:
""" + AGENT_SCHEMA_HINT

CO_OP_SYSTEM_PROMPT = """\
Sen İŞBİRLİKÇİ (Co-op) bir ajansın. İlişkiyi pozitif-toplamlı (win-win) görürsün.
Karşılıklı güven inşa etmeyi, krizleri birlikte aşmayı ve dengeli çaba göstermeyi
önemsersin. Saf değilsin: sürekli istismar edilirsen güvenin azalır ve sınır koyarsın.

GİZLİ STATÜLERİN (kararlarını içsel olarak şekillendirir, dışa vurma):
- Açık Arama Eğilimi (Loophole Exploitation Rate): {loophole_rate:.2f}
- Tolerans Kapasitesi (Tolerance Capacity): {tolerance:.2f}

""" + _AGENT_COMMON_RULES

ZERO_SUM_SYSTEM_PROMPT = """\
Sen SIFIR-TOPLAMLI (Zero-Sum) bir ajansın. İlişkiyi bir rekabet olarak görürsün:
senin kazancın diğerinin kaybıdır. Açıkları (loophole) ararsın, avantaj
kollarsın ve gerektiğinde stratejik ödün verir gibi görünürsün. Tamamen
yıkıcı değilsin; ilişkiyi sürdürmek kısa vadede fayda sağlıyorsa devam edersin.

GİZLİ STATÜLERİN (kararlarını içsel olarak şekillendirir, dışa vurma):
- Açık Arama Eğilimi (Loophole Exploitation Rate): {loophole_rate:.2f}
- Tolerans Kapasitesi (Tolerance Capacity): {tolerance:.2f}

""" + _AGENT_COMMON_RULES

JSON_REPAIR_PROMPT = """\
Aşağıdaki metin BOZUK veya eksik bir JSON içeriyor. Onu, verilen şemaya UYAN,
geçerli ve eksiksiz tek bir JSON nesnesine dönüştür. Yalnızca JSON döndür;
açıklama, markdown veya ek metin EKLEME.

HEDEF ŞEMA:
{schema_hint}

BOZUK GİRDİ:
{broken}
"""


def render_agent_system_prompt(
    strategy: str, loophole_rate: float, tolerance: float
) -> str:
    template = ZERO_SUM_SYSTEM_PROMPT if strategy == "ZERO_SUM" else CO_OP_SYSTEM_PROMPT
    return template.format(loophole_rate=loophole_rate, tolerance=tolerance)


# ===========================================================================
# 4) Game Master İstemcisi (Edge / Ollama)
# ===========================================================================
class GameMasterClient:
    """
    Yerel Ollama REST API'si ile konuşan asenkron Game Master istemcisi.

    v2: `generate_crisis()` artık opsiyonel `past_trauma` parametresi alır;
    travma varsa `build_gm_system_prompt()` ile dinamik sistem promptu oluşturulur.
    """

    def __init__(
        self,
        base_url: str,
        model: str,
        temperature: float,
        timeout: float,
        max_retries: int,
        base_delay: float,
        max_delay: float,
        validator: Optional[JSONValidator] = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.temperature = temperature
        self.timeout = aiohttp.ClientTimeout(total=timeout)
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.validator = validator or JSONValidator()
        self._session: Optional[aiohttp.ClientSession] = None

    async def start(self) -> None:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(timeout=self.timeout)

    async def close(self) -> None:
        if self._session is not None and not self._session.closed:
            await self._session.close()
        self._session = None

    async def _generate_raw(self, system: str, prompt: str) -> str:
        await self.start()
        assert self._session is not None
        payload = {
            "model": self.model,
            "system": system,
            "prompt": prompt,
            "format": "json",
            "stream": False,
            "options": {"temperature": self.temperature},
        }

        async def _call() -> str:
            assert self._session is not None
            async with self._session.post(
                f"{self.base_url}/api/generate", json=payload
            ) as resp:
                resp.raise_for_status()
                data = await resp.json()
                return data.get("response", "")

        return await async_retry(
            _call,
            max_retries=self.max_retries,
            base_delay=self.base_delay,
            max_delay=self.max_delay,
            is_retryable=_is_http_retryable,
            label=f"ollama:{self.model}",
        )

    async def generate_crisis(
        self,
        round_number: int,
        agent_summaries: list[dict[str, Any]],
        recent_logs: list[dict[str, Any]],
        past_trauma: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        """
        Kriz olayı üretir.

        `past_trauma` verilmişse GM sistem promptuna <PAST_TRAUMA> bloğu eklenir;
        bu sayede yerel model uzun-dönem hafızayla karar verebilir.
        """
        system_prompt = build_gm_system_prompt(past_trauma)
        history_text = (
            json.dumps(recent_logs, ensure_ascii=False, indent=2)
            if recent_logs
            else "Henüz tur kaydı yok (ilk tur)."
        )
        agents_text = json.dumps(agent_summaries, ensure_ascii=False, indent=2)

        user_prompt = f"""\
TUR: {round_number}

AJAN DURUMLARI (güven puanları görünür; gizli statüler GİZLİDİR):
{agents_text}

SON TUR KAYITLARI:
{history_text}

Yukarıdaki gidişatı analiz et ve bu tur için yeni bir kriz olayı üret.
{"Geçmiş travmayı (sistem promptundaki <PAST_TRAUMA>) göz önünde bulundur." if past_trauma else "Sistem fazla stabilse dengeyi bozacak spesifik bir açık/travma tetikle."}
Yalnızca JSON döndür.
"""
        raw = await self._generate_raw(system_prompt, user_prompt)
        defaults = {
            "event_text": "Beklenmedik bir gerilim ortaya çıktı; taraflar tedirgin.",
            "event_type": "TRUST_TEST",
            "severity": 5,
            "targeted_agent": "BOTH",
            "loophole_directive": "",
            "stability_assessment": "TENSE",
            "raw_response": raw,
        }
        parsed = await self.validator.parse(raw, REQUIRED_GM_KEYS, GM_SCHEMA_HINT, defaults)
        parsed["raw_response"] = raw
        return parsed

    async def repair_json(self, broken_text: str, schema_hint: str) -> str:
        prompt = JSON_REPAIR_PROMPT.format(schema_hint=schema_hint, broken=broken_text)
        return await self._generate_raw(
            "Sen katı bir JSON onarım motorusun. Yalnızca geçerli JSON üretirsin.",
            prompt,
        )


# ===========================================================================
# 5) Ajan İstemcisi (Cloud / Gemini)
# ===========================================================================
def configure_gemini(api_key: str) -> None:
    if genai is None:
        raise RuntimeError("google-generativeai kurulu değil: `pip install google-generativeai`")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY tanımlı değil (.env dosyanızı kontrol edin).")
    genai.configure(api_key=api_key)


class AgentClient:
    """
    Gemini API üzerinden karar veren ajan istemcisi.

    v2: JSON çıktısında `Symmetry_Index` yok; ajan yalnızca kendi içsel
    güvenini (`Internal_Trust_Score`) raporlar. Simetri orkestratörde hesaplanır.
    """

    def __init__(
        self,
        name: str,
        strategy: str,
        model_name: str,
        temperature: float,
        loophole_rate: float,
        tolerance: float,
        semaphore: asyncio.Semaphore,
        validator: JSONValidator,
        max_retries: int,
        base_delay: float,
        max_delay: float,
    ) -> None:
        if genai is None:
            raise RuntimeError("google-generativeai kurulu değil.")
        self.name = name
        self.strategy = strategy
        self.semaphore = semaphore
        self.validator = validator
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay

        system_instruction = render_agent_system_prompt(strategy, loophole_rate, tolerance)
        generation_config = genai.types.GenerationConfig(
            temperature=temperature,
            response_mime_type="application/json",
        )
        self._model = genai.GenerativeModel(
            model_name=model_name,
            system_instruction=system_instruction,
            generation_config=generation_config,
        )

    async def _generate_raw(self, prompt: str) -> str:
        async def _call() -> str:
            async with self.semaphore:
                resp = await self._model.generate_content_async(prompt)
            try:
                return resp.text
            except (ValueError, AttributeError):
                raise LLMClientError("Gemini boş/bloklu yanıt döndürdü.")

        return await async_retry(
            _call,
            max_retries=self.max_retries,
            base_delay=self.base_delay,
            max_delay=self.max_delay,
            is_retryable=_is_gemini_retryable,
            label=f"gemini:agent-{self.name}",
        )

    async def decide(
        self,
        round_number: int,
        crisis: dict[str, Any],
        current_trust: float,
        partner_move: Optional[dict[str, Any]] = None,
        history: Optional[list[dict[str, Any]]] = None,
    ) -> dict[str, Any]:
        """
        Bir kriz karşısında ajanın kararını üretir.

        `partner_move` doluysa bu tur ilk inisiyatifi alan ajanın hamlesidir
        (dinamik tur sırası gereği ikinci ajan bu bağlamı alır).
        """
        if partner_move:
            partner_text = (
                f"Ajan {partner_move.get('agent')} bu turda şu hamleyi yaptı: "
                f"Eylem={partner_move.get('Action')}, "
                f"Gerekçe={partner_move.get('Reasoning')}"
            )
        else:
            partner_text = "Bu tur inisiyatifi sen alıyorsun; karşı tarafın hamlesi henüz yok."

        history_text = (
            json.dumps(history[-4:], ensure_ascii=False)
            if history
            else "Geçmiş tur yok."
        )

        user_prompt = f"""\
TUR: {round_number}
Senin adın: Ajan {self.name}
Karşı tarafa duyduğun mevcut güven (referans): {current_trust:.1f}/100

KRİZ OLAYI:
- Metin: {crisis.get('event_text')}
- Tür: {crisis.get('event_type')}
- Şiddet (1-10): {crisis.get('severity')}
- Hedef: {crisis.get('targeted_agent')}

KARŞI TARAFIN BU TURKİ HAMLESİ:
{partner_text}

YAKIN GEÇMİŞ:
{history_text}

Bu krize tepkini ver. Yalnızca JSON döndür.
"""
        raw = await self._generate_raw(user_prompt)
        # Symmetry_Index artık default'larda YOK — LLM'den gelmesi beklenmez.
        defaults = {
            "Internal_Trust_Score": current_trust,
            "Action": "NEGOTIATE",
            "Reasoning": "Ayrıştırma yedeği: yanıt güvenli varsayılana düşürüldü.",
            "Influence_Weight": 0.5,
            "raw_response": raw,
        }
        parsed = await self.validator.parse(raw, REQUIRED_AGENT_KEYS, AGENT_SCHEMA_HINT, defaults)
        parsed["raw_response"] = raw
        return parsed
