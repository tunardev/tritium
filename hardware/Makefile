YOSYS ?= yosys
NEXTPNR ?= nextpnr-ecp5
FREQUENCY ?= 50
BUILD := build
PACKAGE := rtl/tritium_package.sv
SOURCES := $(PACKAGE) $(filter-out $(PACKAGE),$(wildcard rtl/*.sv))

.PHONY: synthesis timing clean

synthesis: $(BUILD)/tritium.json

timing: $(BUILD)/timing.json

$(BUILD)/tritium.json: $(SOURCES) $(wildcard rtl/*.svh)
	mkdir -p $(BUILD)
	$(YOSYS) -m slang -q -l $(BUILD)/yosys.log -p "read_slang -I rtl $(SOURCES) --top tritium_gpu; synth_ecp5 -nowidelut -top tritium_gpu -json $@; tee -o $(BUILD)/utilization.txt stat"

$(BUILD)/timing.json: $(BUILD)/tritium.json
	$(NEXTPNR) --85k --package CABGA381 --out-of-context --timing-allow-fail --freq $(FREQUENCY) --json $< --report $@ -l $(BUILD)/nextpnr.log

clean:
	rm -rf $(BUILD)
