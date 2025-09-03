import os
from flask import Flask, request, jsonify
from flask_cors import CORS
import mysql.connector
import unicodedata
from dotenv import load_dotenv

load_dotenv()


def get_db_conn():
    return mysql.connector.connect(
        host=os.environ.get('DB_HOST', 'localhost'),
        user=os.environ.get('DB_USER', 'root'),
        password=os.environ.get('DB_PASSWORD', ''),
        database=os.environ.get('DB_NAME', 'face_recognition'),
        autocommit=True,
        charset='utf8mb4',
        use_unicode=True,
        collation='utf8mb4_unicode_ci',
    )


app = Flask(__name__)
CORS(app)
# Ensure Unicode characters are returned as-is in JSON
app.config['JSON_AS_ASCII'] = False


@app.get('/api/health')
def health():
    return jsonify({'status': 'ok'})


# Helpers for name normalization across endpoints
_TURKISH_MAP = str.maketrans({
    'ç': 'c', 'Ç': 'c',
    'ğ': 'g', 'Ğ': 'g',
    'ı': 'i', 'I': 'i', 'İ': 'i',
    'ö': 'o', 'Ö': 'o',
    'ş': 's', 'Ş': 's',
    'ü': 'u', 'Ü': 'u',
})

def strip_titles(name: str) -> str:
    tokens = (name or '').replace(',', ' ').split()
    title_set = {'dr', 'dr.', 'uzm', 'uzm.', 'prof', 'prof.', 'prof.dr', 'prof.dr.', 'doc', 'doc.', 'doç', 'doç.', 'md', 'm.d.'}
    filtered = [t for t in tokens if t.lower() not in title_set]
    return ' '.join(filtered).strip()

def _normalize_password_source(text: str) -> str:
    # Remove diacritics, transliterate Turkish letters, remove spaces, alnum only, lowercase
    if not text:
        return ''
    nfkd = unicodedata.normalize('NFD', text)
    without_marks = ''.join(ch for ch in nfkd if not unicodedata.combining(ch))
    transliterated = without_marks.translate(_TURKISH_MAP)
    return ''.join(ch for ch in transliterated if ch.isalnum()).lower()


@app.get('/api/medicines')
def list_medicines():
    conn = get_db_conn()
    cur = conn.cursor(dictionary=True)
    try:
        cur.execute("""
            SELECT id, name, active_ingredient, dosage, form
            FROM medicines
            ORDER BY name ASC
        """)
        rows = cur.fetchall()
        return jsonify(rows)
    finally:
        cur.close()
        conn.close()


@app.get('/api/doctors')
def list_doctors():
    # Pull distinct doctor names from prescriptions table and deduplicate by normalized form
    try:
        conn = get_db_conn()
        cur = conn.cursor()
        cur.execute("""
            SELECT DISTINCT doctor_name
            FROM prescriptions
            WHERE doctor_name IS NOT NULL AND doctor_name <> ''
            ORDER BY doctor_name ASC
        """)
        rows = cur.fetchall()
        norm_to_display = {}
        for r in rows:
            raw = (r[0] or '').strip()
            base = strip_titles(raw)
            norm = _normalize_password_source(base)
            if not norm:
                continue
            # Keep original casing from DB to preserve Turkish letters
            display = f"Dr. {base}"
            norm_to_display.setdefault(norm, display)
        data = [{'id': v.replace('Dr. ', ''), 'full_name': v} for v in norm_to_display.values()]
        return jsonify(data)
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    finally:
        try:
            cur.close()
            conn.close()
        except Exception:
            pass


@app.post('/api/doctor_login')
def doctor_login():
    data = request.get_json(force=True) or {}
    doctor_name_input = (data.get('doctor_name') or '').strip()
    password = data.get('password') or ''
    if not doctor_name_input or not password:
        return jsonify({'error': 'doctor_name and password required'}), 400

    # Normalize: remove titles like Dr., spaces, accents; lowercase
    full = strip_titles(doctor_name_input)
    # Accept master password from env if provided
    master = os.environ.get('DOCTOR_MASTER_PASSWORD', '')
    if master and password == master:
        return jsonify({'doctor': {'id': full, 'full_name': full}})
    expected = _normalize_password_source(full)
    provided = _normalize_password_source(password)

    # Accept multiple reasonable variants of the full name
    tokens = [t for t in unicodedata.normalize('NFD', full).replace("-", " ").replace(".", " ").split() if t]
    variants = set()
    if tokens:
        variants.add(_normalize_password_source(''.join(tokens)))
        variants.add(_normalize_password_source(' '.join(tokens)))
        variants.add(_normalize_password_source(''.join(ch for ch in full if ch.isalpha())))
        if len(tokens) >= 2:
            variants.add(_normalize_password_source(tokens[0] + tokens[-1]))
    variants.add(expected)

    if provided not in variants:
        return jsonify({'error': 'wrong_password', 'message': 'Invalid credentials'}), 401

    return jsonify({'doctor': {'id': full, 'full_name': full}})


@app.get('/api/patients')
def list_patients():
    conn = get_db_conn()
    cur = conn.cursor(dictionary=True)
    cur.execute("SELECT tc_kimlik as tc, name as name, surname as surname, birth_date FROM patients")
    rows = cur.fetchall()
    cur.close()
    conn.close()
    # Build full name
    for r in rows:
        r['full_name'] = f"{r.get('name','')} {r.get('surname','')}".strip()
    return jsonify(rows)


@app.get('/api/prescriptions')
def list_prescriptions():
    doctor_name = request.args.get('doctor_name')
    patient_tc = request.args.get('patient_tc')
    conn = get_db_conn()
    cur = conn.cursor(dictionary=True)
    base = "SELECT id, prescription_number, patient_tc, doctor_name, hospital_name, diagnosis, start_date, end_date, status, created_at FROM prescriptions"
    params = []
    where = []
    if doctor_name:
        where.append("doctor_name = %s")
        params.append(doctor_name)
    if patient_tc:
        where.append("patient_tc = %s")
        params.append(patient_tc)
    if where:
        base += " WHERE " + " AND ".join(where)
    base += " ORDER BY created_at DESC"
    cur.execute(base, params)
    pres = cur.fetchall()

    # fetch medicines per prescription
    if pres:
        ids = tuple([p['id'] for p in pres])
        q = f"""SELECT pm.prescription_id, m.name, m.active_ingredient, m.dosage, pm.dosage_instruction
                FROM prescription_medicines pm
                JOIN medicines m ON m.id = pm.medicines_id
                WHERE pm.prescription_id IN ({','.join(['%s']*len(ids))})
        """
        cur.execute(q, ids)
        meds = cur.fetchall()
        id_to_meds = {}
        for m in meds:
            id_to_meds.setdefault(m['prescription_id'], []).append({
                'name': m['name'],
                'active_ingredient': m['active_ingredient'],
                'dosage': m['dosage'],
                'dosage_instruction': m['dosage_instruction'],
        })
        for p in pres:
            p['medicines'] = id_to_meds.get(p['id'], [])

    cur.close()
    conn.close()
    return jsonify(pres)


@app.post('/api/prescriptions')
def create_prescription():
    data = request.get_json(force=True)
    required = ['patient_tc', 'doctor_name', 'hospital_name', 'diagnosis', 'start_date', 'end_date', 'medicines']
    if not all(k in data and data[k] for k in required):
        return jsonify({'error': 'Missing required fields'}), 400
    if not isinstance(data['medicines'], list) or len(data['medicines']) == 0:
        return jsonify({'error': 'At least one medicine required'}), 400

    # Accept both YYYY-MM-DD and DD.MM.YYYY for dates
    def normalize_date(val: str) -> str:
        if not isinstance(val, str):
            return val
        v = val.strip()
        if '.' in v:
            try:
                d, m, y = v.split('.')
                d = d.zfill(2)
                m = m.zfill(2)
                return f"{y}-{m}-{d}"
            except Exception:
                return v
        return v

    data['start_date'] = normalize_date(data.get('start_date', ''))
    data['end_date'] = normalize_date(data.get('end_date', ''))

    try:
        conn = get_db_conn()
        cur = conn.cursor()
        # Insert prescription
        cur.execute(
            """
            INSERT INTO prescriptions (prescription_number, patient_tc, doctor_name, hospital_name, diagnosis, start_date, end_date, status, created_at)
            VALUES (%s, %s, %s, %s, %s, %s, %s, 'active', NOW())
            """,
            (
                data.get('prescription_number') or f"RX{int(__import__('time').time()*1000)}",
                data['patient_tc'],
                data['doctor_name'],
                data['hospital_name'],
                data['diagnosis'],
                data['start_date'],
                data['end_date'],
            )
        )
        prescription_id = cur.lastrowid

        # Insert medicines
        for m in data['medicines']:
            cur.execute(
                """
                INSERT INTO prescription_medicines (prescription_id, medicines_id, dosage_instruction)
                VALUES (%s, %s, %s)
                """,
                (
                    prescription_id,
                    m.get('id'),
                    m.get('dosage_instruction', ''),
                )
            )
            # dispensing_log kaydı (reçete yazımında başlangıç kaydı)
            try:
                cur.execute(
                    """
                    INSERT INTO dispensing_log (medicine_id, quantity, dispensed_at, patient_tc)
                    VALUES (%s, %s, NOW(), %s)
                    """,
                    (
                        m.get('id'),
                        int(m.get('quantity', 1) or 1),
                        data['patient_tc'],
                    )
                )
            except Exception:
                # dispensing_log optional; hata olsa da reçete kaydını sürdür
                pass

        conn.commit()
        return jsonify({'id': prescription_id}), 201
    except Exception as e:
        try:
            conn.rollback()
        except Exception:
            pass
        return jsonify({'error': 'insert_failed', 'message': str(e)}), 500
    finally:
        try:
            cur.close()
            conn.close()
        except Exception:
            pass


if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000, debug=False)


