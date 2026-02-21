cat > dedup_full.py << 'ENDOFFILE'
import os, logging, time, psycopg, pandas as pd, re, requests
from psycopg.rows import dict_row
from anthropic import Anthropic
from anthropic._exceptions import RateLimitError
from tenacity import retry, wait_random_exponential, stop_after_attempt, retry_if_exception_type
from concurrent.futures import ThreadPoolExecutor, as_completed
from difflib import SequenceMatcher
from bs4 import BeautifulSoup
from typing import Dict, List

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

client = Anthropic(api_key=os.getenv("CLAUDE_API_KEY"))
CLAUDE_MODEL = "claude-sonnet-4-5-20250929"
MAX_WORKERS = 15
RATE_LIMIT_DELAY = 0.15

def get_db():
    return psycopg.connect(os.environ.get('DATABASE_URL'), row_factory=dict_row)

def load_articles():
    conn = get_db()
    cur = conn.cursor()
    cur.execute('''SELECT id, published_date, news_title, link, "Source", competitor_tagging, 
                   sbu_tagging, category_tag, summary, scraped_content, rank_score 
                   FROM processed_articles ORDER BY published_date DESC''')
    results = cur.fetchall()
    cur.close()
    conn.close()
    if not results:
        return pd.DataFrame()
    df = pd.DataFrame(results)
    df = df.rename(columns={'news_title': 'News Title', 'published_date': 'Published Date', 
                            'competitor_tagging': 'Competitor'})
    return df

def delete_ids(ids):
    if not ids:
        logging.info("No duplicates to delete")
        return
    conn = get_db()
    cur = conn.cursor()
    for i in range(0, len(ids), 100):
        batch = ids[i:i+100]
        placeholders = ','.join(['%s'] * len(batch))
        cur.execute(f"DELETE FROM processed_articles WHERE id IN ({placeholders})", batch)
    conn.commit()
    cur.close()
    conn.close()
    logging.info(f"✅ Deleted {len(ids)} duplicates from database")

def extract_numbers(text):
    numbers = []
    indian_format = re.findall(r'(?:rs|₹|inr)\.?\s*([\d,]+)', text, re.IGNORECASE)
    for match in indian_format:
        try:
            num = float(match.replace(',', ''))
            if num > 10_000_000:
                num = num / 10_000_000
            elif num > 100_000:
                num = num / 10_000_000
            numbers.append(round(num, 2))
        except:
            pass
    patterns = [
        (r'(?:rs|₹|inr)?\.?\s*(\d+(?:[,.]\d+)*)\s*(?:crore|cr)', 1.0),
        (r'(?:rs|₹|inr)?\.?\s*(\d+(?:[,.]\d+)*)\s*(?:lakh|lac)', 0.01),
        (r'(\d+(?:[,.]\d+)*)\s*(?:million|mn)', 8.5),
    ]
    for pattern, multiplier in patterns:
        for match in re.findall(pattern, text, re.IGNORECASE):
            try:
                numbers.append(round(float(match.replace(',', '')) * multiplier, 2))
            except:
                pass
    return numbers

def similar_numbers(n1, n2):
    if not n1 or not n2:
        return False
    for a in n1:
        for b in n2:
            if a > 0 and b > 0 and abs(a - b) / max(a, b) * 100 < 10:
                return True
    return False

def core_match(t1, t2):
    stop = {'a','an','the','and','or','but','in','on','at','to','for','of','with','by','from',
            'as','is','was','are','were','been','be','have','has','had','worth','order','contract'}
    w1 = set(re.findall(r'\b\w+\b', t1.lower())) - stop
    w2 = set(re.findall(r'\b\w+\b', t2.lower())) - stop
    if not w1 or not w2:
        return False
    return (len(w1 & w2) / min(len(w1), len(w2)) * 100) >= 60

def phase1_dedup(df):
    logging.info("\n🔍 Phase 1: String-based deduplication...")
    if df.empty:
        return df, []
    df = df.reset_index(drop=True)
    to_drop = set()
    ids = []
    seen = {}
    for i in range(len(df)):
        t = str(df.iloc[i]['News Title']).lower().strip()
        if t in seen:
            to_drop.add(i)
            ids.append(df.iloc[i]['id'])
        else:
            seen[t] = i
    logging.info(f"   Exact: {len(to_drop)}")
    
    for i in range(len(df)):
        if i in to_drop:
            continue
        t1 = str(df.iloc[i]['News Title']).lower()
        d1 = df.iloc[i]['Published Date']
        c1 = str(df.iloc[i].get('Competitor') or '').lower()
        n1 = extract_numbers(t1)
        r1 = df.iloc[i].get('rank_score', 0) or 0
        
        for j in range(i+1, min(i+100, len(df))):
            if j in to_drop:
                continue
            t2 = str(df.iloc[j]['News Title']).lower()
            d2 = df.iloc[j]['Published Date']
            c2 = str(df.iloc[j].get('Competitor') or '').lower()
            r2 = df.iloc[j].get('rank_score', 0) or 0
            
            try:
                dd = abs((d1 - d2).days)
            except:
                dd = 0
            if dd > 3:
                continue
            
            n2 = extract_numbers(t2)
            sim = SequenceMatcher(None, t1, t2).ratio()
            same_c = c1 == c2 and c1 not in ('', '-')
            same_v = similar_numbers(n1, n2)
            cm = core_match(t1, t2)
            
            dup = False
            if sim > 0.85:
                dup = True
            elif same_c and same_v and dd <= 1:
                dup = True
            elif same_c and cm and dd <= 2:
                dup = True
            
            if dup:
                if r2 >= r1:
                    to_drop.add(i)
                    ids.append(df.iloc[i]['id'])
                    break
                else:
                    to_drop.add(j)
                    ids.append(df.iloc[j]['id'])
    
    logging.info(f"   Fuzzy: {len(to_drop) - len(seen) + len(df)}")
    logging.info(f"   Phase 1 removes: {len(to_drop)}")
    return df.drop(index=list(to_drop)).reset_index(drop=True), ids

CATEGORY_FIELDS = {
    "order wins": ["company", "client_or_authority", "contract_value_crore", "scope", "location"],
    "financial": ["company", "period", "revenue_crore", "profit_crore"],
    "stock market": ["company", "price_movement_percent", "trigger_event"],
}

FINGERPRINT_PROMPT = "Extract key facts as JSON. Return ONLY JSON, no explanation."

def scrape(url):
    try:
        r = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'}, timeout=10)
        r.raise_for_status()
        soup = BeautifulSoup(r.content, 'html.parser')
        for e in soup(["script", "style", "nav", "footer", "aside", "header", "iframe"]):
            e.decompose()
        text = ' '.join(soup.get_text(separator=' ', strip=True).split())
        return text[:3000]
    except:
        return ""

@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(3), 
       retry=retry_if_exception_type(RateLimitError), reraise=True)
def fingerprint(title, content, category):
    fields = CATEGORY_FIELDS.get(category.lower(), ["company", "event_type", "value"])
    fields_desc = "\n".join([f'"{f}": null' for f in fields])
    prompt = f"Title: {title}\nContent: {content[:2000]}\nCategory: {category}\n\nExtract:\n{{\n{fields_desc}\n}}"
    
    try:
        resp = client.messages.create(model=CLAUDE_MODEL, max_tokens=300, temperature=0,
                                      system=FINGERPRINT_PROMPT, messages=[{"role":"user","content":prompt}])
        raw = resp.content[0].text.strip()
        raw = re.sub(r'^```json\s*', '', raw)
        raw = re.sub(r'^```\s*', '', raw)
        raw = re.sub(r'\s*```$', '', raw)
        match = re.search(r'\{[\s\S]*\}', raw)
        if match:
            return eval(match.group(0))
        return {}
    except:
        return {}

def fp_match(fp1, fp2, cat):
    if not fp1 or not fp2:
        return False
    
    def norm(v):
        return str(v).lower().strip() if v else None
    
    def val_sim(v1, v2):
        try:
            n1, n2 = float(v1), float(v2)
            if n1 == 0 and n2 == 0:
                return True
            if n1 == 0 or n2 == 0:
                return False
            return abs(n1 - n2) / max(n1, n2) <= 0.10
        except:
            return False
    
    def comp_match(c1, c2):
        if not c1 or not c2:
            return False
        c1, c2 = norm(c1), norm(c2)
        if c1 == c2 or c1 in c2 or c2 in c1:
            return True
        return SequenceMatcher(None, c1, c2).ratio() > 0.80
    
    cat = cat.lower()
    if cat == "order wins":
        return (comp_match(fp1.get('company'), fp2.get('company')) and
                (val_sim(fp1.get('contract_value_crore'), fp2.get('contract_value_crore')) if fp1.get('contract_value_crore') and fp2.get('contract_value_crore') else True))
    elif cat == "financial":
        return (comp_match(fp1.get('company'), fp2.get('company')) and
                (norm(fp1.get('period')) == norm(fp2.get('period')) or not fp1.get('period')))
    else:
        m, t = 0, 0
        for k in fp1:
            if fp1[k] and fp2.get(k):
                t += 1
                if norm(fp1[k]) == norm(fp2[k]):
                    m += 1
        return (m / t) >= 0.6 if t > 0 else False

def phase2_dedup(df):
    logging.info("\n🤖 Phase 2: LLM semantic deduplication...")
    if df.empty or len(df) <= 1:
        return df, []
    if 'rank_score' not in df.columns:
        df['rank_score'] = 0
    df = df.reset_index(drop=True)
    ids = []
    
    logging.info(f"   📥 Scraping {len(df)} articles...")
    contents = {}
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futures = {ex.submit(scrape, row['link']): i for i, row in df.iterrows() if pd.notna(row.get('link'))}
        for f in as_completed(futures):
            contents[futures[f]] = f.result()
    
    logging.info(f"   🔑 Extracting fingerprints...")
    fps = {}
    for i, row in df.iterrows():
        fps[i] = fingerprint(str(row['News Title']), contents.get(i, str(row.get('scraped_content',''))), 
                            str(row.get('category_tag', 'order wins')))
        logging.info(f"   [{i+1}/{len(df)}] {str(row['News Title'])[:60]}...")
        time.sleep(RATE_LIMIT_DELAY)
    
    logging.info("\n   🔍 Comparing fingerprints...")
    to_drop = set()
    groups = {}
    for i, row in df.iterrows():
        comps = [c.strip() for c in str(row.get('Competitor') or 'General').split(',') if c.strip() != '-']
        if not comps:
            comps = ['General']
        for c in comps:
            if c not in groups:
                groups[c] = []
            groups[c].append(i)
    
    logging.info(f"   📦 {len(groups)} groups")
    compared = set()
    
    for comp, indices in groups.items():
        if len(indices) <= 1:
            continue
        logging.info(f"   👥 {comp}: {len(indices)} articles")
        
        for i in range(len(indices)):
            idx_i = indices[i]
            if idx_i in to_drop:
                continue
            for j in range(i+1, len(indices)):
                idx_j = indices[j]
                if idx_j in to_drop:
                    continue
                pair = (min(idx_i, idx_j), max(idx_i, idx_j))
                if pair in compared:
                    continue
                compared.add(pair)
                
                r1, r2 = df.iloc[idx_i], df.iloc[idx_j]
                try:
                    dd = abs((r1['Published Date'] - r2['Published Date']).days)
                except:
                    dd = 0
                if dd > 3:
                    continue
                
                c1 = str(r1.get('category_tag','')).lower()
                c2 = str(r2.get('category_tag','')).lower()
                if c1 != c2:
                    continue
                
                if fp_match(fps.get(idx_i, {}), fps.get(idx_j, {}), c1):
                    s1 = float(r1.get('rank_score') or 0)
                    s2 = float(r2.get('rank_score') or 0)
                    drop_idx = idx_j if s1 >= s2 else idx_i
                    to_drop.add(drop_idx)
                    ids.append(df.iloc[drop_idx]['id'])
                    logging.info(f"   🗑️  DUPLICATE: '{df.iloc[drop_idx]['News Title'][:60]}'")
    
    logging.info(f"\n   Phase 2 removes: {len(to_drop)}")
    return df.drop(index=list(to_drop)).reset_index(drop=True), ids

logging.info("="*60)
logging.info("DATABASE DEDUPLICATION - TWO-PHASE")
logging.info("="*60)

df = load_articles()
if df.empty:
    logging.info("No articles found")
else:
    init = len(df)
    logging.info(f"📄 Loaded {init} articles")
    
    df, ids1 = phase1_dedup(df)
    logging.info(f"✅ After Phase 1: {len(df)} articles")
    
    df, ids2 = phase2_dedup(df)
    logging.info(f"✅ After Phase 2: {len(df)} articles")
    
    all_ids = ids1 + ids2
    logging.info(f"\n📊 Total duplicates: {len(all_ids)} ({len(all_ids)/init*100:.1f}%)")
    
    if all_ids:
        delete_ids(all_ids)
    
    logging.info("="*60)
    logging.info("✅ Complete!")
    logging.info("="*60)
ENDOFFILE