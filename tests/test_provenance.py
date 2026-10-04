from hydra.provenance import RunStore, make_record


def test_identical_inputs_give_identical_run_id():
    p = {"al_g": 5.0, "naoh_M": 2.0}
    a = make_record(p, seed=1, solver={"rtol": 1e-8})
    b = make_record(p, seed=1, solver={"rtol": 1e-8})
    assert a.run_id == b.run_id
    assert make_record(p, seed=2, solver={"rtol": 1e-8}).run_id != a.run_id


def test_store_roundtrip(tmp_path):
    st = RunStore(tmp_path / "x.sqlite3")
    rec = make_record({"al_g": 5.0}, seed=7)
    st.save(rec)
    assert st.load(rec.run_id) == rec
    assert st.list_ids() == [rec.run_id]
