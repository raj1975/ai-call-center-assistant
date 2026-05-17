from utils.sensitive_data import detect_profanity, detect_sensitive, mask_profanity, mask_sensitive


class TestDetectSensitive:
    def test_ssn_detected(self):
        result = detect_sensitive("My SSN is 523-74-1892")
        assert result["has_sensitive_data"] is True
        assert "PII — Social Security Number" in result["sensitive_data_types"]

    def test_card_number_detected(self):
        result = detect_sensitive("Card number is 4111 1111 1111 1234")
        assert result["has_sensitive_data"] is True
        assert "PCI — card number" in result["sensitive_data_types"]

    def test_cvv_detected(self):
        result = detect_sensitive("CVV 342")
        assert result["has_sensitive_data"] is True
        assert "PCI — CVV / security code" in result["sensitive_data_types"]

    def test_expiry_detected(self):
        result = detect_sensitive("expiry 09/27")
        assert result["has_sensitive_data"] is True
        assert "PCI — card expiry" in result["sensitive_data_types"]

    def test_email_detected(self):
        result = detect_sensitive("email me at user@example.com")
        assert result["has_sensitive_data"] is True
        assert "PII — email address" in result["sensitive_data_types"]

    def test_phone_detected(self):
        result = detect_sensitive("call me at 415-882-3301")
        assert result["has_sensitive_data"] is True
        assert "PII — phone number" in result["sensitive_data_types"]

    def test_mrn_detected(self):
        result = detect_sensitive("medical record number MRN-00445521")
        assert result["has_sensitive_data"] is True
        assert "PHI — medical record number" in result["sensitive_data_types"]

    def test_dob_detected(self):
        result = detect_sensitive("date of birth 07/14/1978")
        assert result["has_sensitive_data"] is True
        assert "PHI — date of birth" in result["sensitive_data_types"]

    def test_phi_term_detected(self):
        result = detect_sensitive("diagnosed with hypertension")
        assert result["has_sensitive_data"] is True
        assert "PHI — medical / health information" in result["sensitive_data_types"]

    def test_insurance_member_id_detected(self):
        result = detect_sensitive("insurance member ID MED-HI-2291847")
        assert result["has_sensitive_data"] is True

    def test_clean_text_not_flagged(self):
        result = detect_sensitive("Hello, how can I help you today?")
        assert result["has_sensitive_data"] is False
        assert result["sensitive_data_types"] == []

    def test_multiple_types_detected(self):
        result = detect_sensitive("SSN 523-74-1892 and card 4111 1111 1111 1234")
        assert len(result["sensitive_data_types"]) >= 2

    def test_empty_string(self):
        result = detect_sensitive("")
        assert result["has_sensitive_data"] is False

    # ── New: standalone / bare-value patterns ─────────────────────────────────

    def test_dob_month_name_detected(self):
        result = detect_sensitive("It is March 14, 1971.")
        assert result["has_sensitive_data"] is True
        assert "PHI — date of birth" in result["sensitive_data_types"]

    def test_dob_month_name_with_ordinal_detected(self):
        result = detect_sensitive("My birthday is July 4th, 1990.")
        assert result["has_sensitive_data"] is True

    def test_expiry_bare_mm_yy_detected(self):
        result = detect_sensitive("09-28")
        assert result["has_sensitive_data"] is True
        assert "PCI — card expiry" in result["sensitive_data_types"]

    def test_expiry_bare_mm_slash_yy_detected(self):
        result = detect_sensitive("09/28")
        assert result["has_sensitive_data"] is True

    def test_full_date_not_flagged_as_expiry(self):
        # 07/14/1978 should NOT be detected as a card expiry (MM/YY)
        result = detect_sensitive("date of birth 07/14/1978")
        expiry_hits = [t for t in result["sensitive_data_types"] if "expiry" in t.lower()]
        assert len(expiry_hits) == 0

    def test_no_duplicate_labels(self):
        # Two expiry patterns both match — label should appear only once
        result = detect_sensitive("expiry 09/28")
        expiry_hits = [t for t in result["sensitive_data_types"] if "expiry" in t.lower()]
        assert len(expiry_hits) == 1

    def test_dob_month_name_no_comma(self):
        # "September 3rd 1979" — ordinal, no comma before year
        result = detect_sensitive("September 3rd 1979")
        assert result["has_sensitive_data"] is True
        assert "PHI — date of birth" in result["sensitive_data_types"]

    def test_phone_bare_response_detected(self):
        result = detect_sensitive("650-441-8822")
        assert result["has_sensitive_data"] is True
        assert "PII — phone number" in result["sensitive_data_types"]


class TestDetectProfanity:
    def test_bullshit_detected(self):
        assert detect_profanity("This is complete bullshit") is True

    def test_damn_detected(self):
        assert detect_profanity("Three damn days!") is True

    def test_crap_detected(self):
        assert detect_profanity("This whole thing is crap") is True

    def test_pissed_detected(self):
        assert detect_profanity("I am so pissed off") is True

    def test_clean_text_not_flagged(self):
        assert detect_profanity("I am very frustrated with the service") is False

    def test_case_insensitive(self):
        assert detect_profanity("This is BULLSHIT") is True

    def test_empty_string(self):
        assert detect_profanity("") is False

    def test_partial_word_not_flagged(self):
        # "classic" contains "ass" but should not be flagged as standalone word
        assert detect_profanity("That is a classic problem") is False


class TestMaskSensitive:
    def test_ssn_masked_last_four_visible(self):
        result = mask_sensitive("My SSN is 523-74-1892 and I need help")
        assert "523-74-1892" not in result
        assert "###-##-1892" in result

    def test_email_masked(self):
        result = mask_sensitive("email me at user@example.com please")
        assert "user@example.com" not in result
        assert "####" in result

    def test_card_number_masked(self):
        result = mask_sensitive("card number is 4111 1111 1111 1234")
        assert "4111" not in result
        assert "####" in result

    def test_phone_masked(self):
        result = mask_sensitive("call me at 415-882-3301 anytime")
        assert "415-882-3301" not in result
        assert "####" in result

    def test_clean_text_unchanged(self):
        text = "Hello, how can I help you today?"
        assert mask_sensitive(text) == text

    def test_surrounding_text_preserved(self):
        result = mask_sensitive("My name is John and my SSN is 523-74-1892")
        assert "My name is John" in result
        assert "523-74-1892" not in result

    def test_multiple_values_masked(self):
        result = mask_sensitive("SSN 523-74-1892 email user@example.com")
        assert "523-74-1892" not in result
        assert "user@example.com" not in result

    def test_dob_month_name_masked_year_preserved(self):
        result = mask_sensitive("It is March 14, 1971.")
        assert "March 14" not in result
        assert "1971" in result
        assert "[DOB: 1971]" in result

    def test_dob_numeric_masked_year_preserved(self):
        result = mask_sensitive("date of birth 07/14/1978")
        assert "07/14" not in result
        assert "1978" in result
        assert "##/##/1978" in result

    def test_expiry_bare_masked(self):
        result = mask_sensitive("09-28")
        assert "09-28" not in result
        assert "####" in result

    # ── Context-aware masking (bare response after agent question) ────────────

    def test_cvv_bare_response_masked(self):
        transcript = "Agent: And the security code on the back?\n\nCustomer: 316."
        result = mask_sensitive(transcript)
        assert "316" not in result
        assert "####" in result

    def test_expiry_bare_response_masked(self):
        transcript = "Agent: Got it. And the expiry date?\n\nCustomer: 09-28."
        result = mask_sensitive(transcript)
        assert "09-28" not in result

    def test_dob_bare_response_year_preserved(self):
        transcript = "Agent: Can I verify your date of birth?\n\nCustomer: It is March 14, 1971."
        result = mask_sensitive(transcript)
        assert "March 14" not in result
        assert "1971" in result

    def test_dob_numeric_bare_response_year_preserved(self):
        transcript = "Agent: Can I verify your date of birth?\n\nCustomer: 07/14/1978."
        result = mask_sensitive(transcript)
        assert "07/14" not in result
        assert "1978" in result

    def test_non_sensitive_number_not_masked_without_context(self):
        transcript = "Agent: The refund is $180.\n\nCustomer: That sounds good."
        result = mask_sensitive(transcript)
        assert "180" in result

    def test_full_audio_transcript_scenario(self):
        transcript = (
            "Agent: Can I verify your date of birth?\n\n"
            "Customer: It is March 14, 1971.\n\n"
            "Agent: And the expiry date?\n\n"
            "Customer: 09-28.\n\n"
            "Agent: And the security code on the back?\n\n"
            "Customer: 316."
        )
        result = mask_sensitive(transcript)
        assert "March 14" not in result
        assert "1971" in result
        assert "09-28" not in result
        assert "316" not in result

    def test_dob_no_comma_year_preserved(self):
        result = mask_sensitive("Customer: September 3rd 1979.")
        assert "September 3rd" not in result
        assert "1979" in result
        assert "[DOB: 1979]" in result

    def test_phone_bare_response_masked(self):
        result = mask_sensitive("Customer: 650-441-8822.")
        assert "650-441-8822" not in result
        assert "####" in result

    def test_full_sensitive_mp3_scenario(self):
        transcript = (
            "Agent: Can I verify your Social Security Number?\n\n"
            "Customer: It is 412-77-9031.\n\n"
            "Agent: And your date of birth for verification?\n\n"
            "Customer: September 3rd 1979.\n\n"
            "Agent: And your medical record number?\n\n"
            "Customer: MRN-00774412.\n\n"
            "Agent: Can I take a card number?\n\n"
            "Customer: The number is 5486 7233 9911 0047.\n\n"
            "Agent: And the expiry date on that card?\n\n"
            "Customer: 03-27.\n\n"
            "Agent: And the security code?\n\n"
            "Customer: 881.\n\n"
            "Agent: And your best callback number?\n\n"
            "Customer: 650-441-8822."
        )
        result = mask_sensitive(transcript)
        # SSN: last 4 visible
        assert "###-##-9031" in result
        # DOB: year preserved
        assert "1979" in result
        assert "September 3rd" not in result
        # Card, CVV, expiry fully masked
        assert "5486" not in result
        assert "881" not in result
        assert "03-27" not in result
        # Phone masked
        assert "650-441-8822" not in result


class TestMaskProfanity:
    def test_bullshit_masked(self):
        result = mask_profanity("This is complete bullshit")
        assert "bullshit" not in result
        assert "####" in result

    def test_case_insensitive_masking(self):
        result = mask_profanity("This is BULLSHIT")
        assert "BULLSHIT" not in result
        assert "####" in result

    def test_surrounding_text_preserved(self):
        result = mask_profanity("I am very angry and this is crap service")
        assert "I am very angry and this is" in result
        assert "crap" not in result

    def test_clean_text_unchanged(self):
        text = "I am frustrated with the wait time"
        assert mask_profanity(text) == text

    def test_multiple_words_masked(self):
        result = mask_profanity("This is bullshit and I am pissed")
        assert "bullshit" not in result
        assert "pissed" not in result

    def test_damn_masked(self):
        result = mask_profanity("Three damn days!")
        assert "damn" not in result
        assert "####" in result
