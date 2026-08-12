#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <memory>

#include "UserInterface.h"
#include "core/Context.h"
#include "core/Encode.h"

using word = uint64_t;
using Ct = cheddar::Ciphertext<word>;
using EvkMap = cheddar::EvkMap<word>;
using UI = cheddar::UserInterface<word>;

void chebyshev_cube__configure(std::shared_ptr<cheddar::Context<word>>& ctx,
                               std::unique_ptr<UI>& ui);
void chebyshev_cube__encrypt_inputs(cheddar::Context<word>* ctx, UI* ui,
                                    float input[8], Ct out[1]);
void chebyshev_cube__evaluate(cheddar::Context<word>* ctx,
                              const EvkMap& evk_map, const Ct input[1],
                              Ct out[1]);
void chebyshev_cube__decrypt_outputs(cheddar::Context<word>* ctx, UI* ui,
                                     const Ct input[1], float out[8]);

TEST(CheddarPipelineEvalChebyshevE2E, MatchesPlaintextDegreeFive) {
  float input[8] = {-0.9F, -0.7F, -0.25F, 0.0F, 0.1F, 0.35F, 0.65F, 0.9F};
  std::shared_ptr<cheddar::Context<word>> ctx;
  std::unique_ptr<UI> ui;
  chebyshev_cube__configure(ctx, ui);
  ASSERT_NE(ctx, nullptr);
  ASSERT_NE(ui, nullptr);

  const EvkMap& evk_map = ui->GetEvkMap();
  Ct encrypted[1];
  Ct evaluated[1];
  chebyshev_cube__encrypt_inputs(ctx.get(), ui.get(), input, encrypted);
  chebyshev_cube__evaluate(ctx.get(), evk_map, encrypted, evaluated);
  EXPECT_EQ(ctx->param_.NPToLevel(encrypted[0].GetNP()), 3);
  EXPECT_EQ(ctx->param_.NPToLevel(evaluated[0].GetNP()), 0);

  float actual[8];
  chebyshev_cube__decrypt_outputs(ctx.get(), ui.get(), evaluated, actual);
  for (int i = 0; i < 8; ++i) {
    ASSERT_TRUE(std::isfinite(actual[i]));
    EXPECT_NEAR(actual[i], std::pow(input[i], 5), 1e-3) << i;
  }
}
