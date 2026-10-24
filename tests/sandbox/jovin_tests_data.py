"""HTTP framing and FITS-size arithmetic only; no network or full-data claim."""
from solution import fits_data_bytes, transfer_layout


def test_range_offset_length_and_total_are_checked():
    assert transfer_layout(206, {"Content-Range": "bytes 100-199/500", "Content-Length": "100"}, 100, 500) == (100, 100, 500)
    bad = [({"Content-Range": "bytes 99-199/500"}, 100, 500),
           ({"Content-Range": "bytes 100-199/501"}, 100, 500),
           ({"Content-Range": "bytes 100-500/500"}, 100, 500),
           ({"Content-Range": "bytes 100-199/*"}, 100, None),
           ({"Content-Range": "bytes 100-199/500", "Content-Length": "99"}, 100, 500)]
    for headers, offset, total in bad:
        try:
            transfer_layout(206, headers, offset, total)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid partial body accepted")


def test_ignored_range_restarts_and_unknown_length_stays_unknown():
    assert transfer_layout(200, {"Content-Length": "500"}, 100, 500) == (0, 500, 500)
    assert transfer_layout(200, {}, 0) == (0, None, None)


def test_compressed_and_invalid_responses_fail_closed():
    for status, headers in [(200, {"Content-Encoding": "gzip"}), (200, {"Content-Length": "-1"}),
                            (200, {"Content-Length": "10,10"}), (206, {}), (204, {}),
                            (200, {"Content-Range": "bytes 0-4/5"})]:
        try:
            transfer_layout(status, headers, 0)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid framing accepted")


def test_fits_layout_handles_empty_primary_image_and_binary_table_heap():
    assert fits_data_bytes({"BITPIX": "8", "NAXIS": "0"}) == 0
    assert fits_data_bytes({"BITPIX": "-64", "NAXIS": "2", "NAXIS1": "8", "NAXIS2": "12"}) == 8 * 8 * 12
    assert fits_data_bytes({"BITPIX": "8", "NAXIS": "2", "NAXIS1": "40", "NAXIS2": "50331648",
                            "PCOUNT": "16", "GCOUNT": "1"}) == 40 * 50331648 + 16


def test_fits_invalid_dimensions_are_rejected():
    bad = [{"BITPIX": "7", "NAXIS": "0"}, {"BITPIX": "8", "NAXIS": "-1"},
           {"BITPIX": "8", "NAXIS": "1", "NAXIS1": "-2"},
           {"BITPIX": "8", "NAXIS": "0", "PCOUNT": "-1"},
           {"BITPIX": "8", "NAXIS": "0", "GCOUNT": "0"},
           {"BITPIX": "8", "NAXIS": "0", "GROUPS": "T"}]
    for cards in bad:
        try:
            fits_data_bytes(cards)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid FITS payload dimensions accepted")
