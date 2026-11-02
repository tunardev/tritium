#include <cstdint>
#include <memory>
#include <vector>

#include "Vtritium_gpu.h"
#include "verilated.h"

namespace {

constexpr uint32_t HOST_STATUS = 0;
constexpr uint32_t STATUS_DONE = 2;

struct Harness {
  std::unique_ptr<VerilatedContext> context;
  std::unique_ptr<Vtritium_gpu> model;
  std::vector<uint8_t> global;
  uint32_t latency;
  uint64_t cycles = 0;
  uint32_t countdown = 0;
  uint32_t pending_data = 0;

  Harness(size_t global_bytes, uint32_t latency)
      : context(std::make_unique<VerilatedContext>()),
        model(std::make_unique<Vtritium_gpu>(context.get())),
        global((global_bytes + 3) / 4 * 4, 0),
        latency(latency) {}

  void access() {
    uint32_t address = model->memory_address;
    pending_data = 0;
    if (address + 4 > global.size()) return;
    for (uint32_t byte = 0; byte < 4; byte++) {
      if (model->memory_write && (model->memory_byte_enable >> byte & 1))
        global[address + byte] = model->memory_write_data >> (8 * byte);
      pending_data |= uint32_t(global[address + byte]) << (8 * byte);
    }
  }

  void tick() {
    model->memory_response = 0;
    if (countdown > 0 && --countdown == 0) {
      model->memory_response = 1;
      model->memory_read_data = pending_data;
    }
    model->clock = 0;
    model->eval();
    model->clock = 1;
    model->eval();
    cycles++;
    if (model->memory_request) {
      access();
      countdown = latency;
    }
  }
};

}

double sc_time_stamp() { return 0; }

extern "C" {

Harness* tritium_harness_create(size_t global_bytes, uint32_t latency) {
  auto* harness = new Harness(global_bytes, latency);
  harness->model->reset = 1;
  for (int cycle = 0; cycle < 4; cycle++) harness->tick();
  harness->model->reset = 0;
  return harness;
}

void tritium_harness_destroy(Harness* harness) { delete harness; }

uint8_t* tritium_harness_global(Harness* harness) { return harness->global.data(); }

void tritium_harness_write_register(Harness* harness, uint32_t address, uint32_t value) {
  harness->model->host_address = address;
  harness->model->host_write_data = value;
  harness->model->host_write = 1;
  harness->tick();
  harness->model->host_write = 0;
}

uint32_t tritium_harness_read_register(Harness* harness, uint32_t address) {
  harness->model->host_address = address;
  harness->model->eval();
  return harness->model->host_read_data;
}

uint64_t tritium_harness_run(Harness* harness, uint64_t cycle_limit) {
  uint64_t start = harness->cycles;
  harness->model->host_address = HOST_STATUS;
  while (harness->cycles - start < cycle_limit) {
    harness->tick();
    if (harness->model->host_read_data & STATUS_DONE) break;
  }
  return harness->cycles - start;
}
}
