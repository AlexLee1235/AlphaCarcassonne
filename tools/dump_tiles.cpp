// dump_tiles —— 檢查牌表(tile.hpp 的 all_tiles),再把它寫成 JSON,
// 讓 render_tile_table.py 把每種牌的資料畫在 tiles/<type>.png 上核對。
//
// 檢查的內容在 game/tile_check.hpp(carcassonne_test 也跑同一套):
//   錯誤(會讓 carcassonne_test 失敗):link、半邊、城邊、盾牌不合規則
//   未完成:某個擴充已經開始填,但張數還不等於整盒的張數
//   觀測分不出來:兩種牌在觀測上長得一樣、農田卻不同(見 docs/adding_tiles.md)
//
// 用法: ./dump_tiles [輸出檔=tile_table.json]
// 有錯誤時結束碼是 1,但 JSON 照樣寫出來,方便對著圖找錯。
#include "tile_check.hpp"

#include <cstdio>
#include <string>
#include <vector>

namespace {

void WriteIntArray(FILE *out, const int *values, int count) {
    std::fputc('[', out);
    for (int i = 0; i < count; ++i) {
        std::fprintf(out, "%s%d", i ? ", " : "", values[i]);
    }
    std::fputc(']', out);
}

void WriteJson(const char *path) {
    FILE *out = std::fopen(path, "w");
    if (!out) {
        std::fprintf(stderr, "cannot write %s\n", path);
        return;
    }
    // The bit values, so render_tile_table.py need not copy them.
    std::fprintf(out, "{\n  \"mark_bits\": {\"SH\": %d, \"PR\": %d, \"WI\": %d, \"CL\": %d, \"WH\": %d, \"INN\": %d},\n",
                 MARK_SHIELD, MARK_PRINCESS, MARK_WINE, MARK_CLOTH, MARK_WHEAT, MARK_INN);
    std::fprintf(out,
                 "  \"tile_mark_bits\": {\"monastery\": %d, \"dragon\": %d, \"volcano\": %d, \"portal\": %d, "
                 "\"tunnel\": %d},\n",
                 TILE_MONASTERY, TILE_DRAGON, TILE_VOLCANO, TILE_PORTAL, TILE_TUNNEL);
    std::fprintf(out, "  \"tile_types\": [\n");
    for (int row = 0; row < CANONICAL_TILE_TYPE_COUNT; ++row) {
        const TileBlueprint &bp = all_tiles[row];
        const Tile &tile = bp.tile;
        int links[4], marks[4], fields[HALF_EDGE_COUNT], city_sides[MAX_TILE_FIELDS];
        for (int side = 0; side < 4; ++side) links[side] = tile.link[side];
        for (int side = 0; side < 4; ++side) marks[side] = tile.marks[side];
        for (int e = 0; e < HALF_EDGE_COUNT; ++e) fields[e] = tile.field[e];
        for (int f = 0; f < MAX_TILE_FIELDS; ++f) city_sides[f] = tile.field_city_sides[f];
        std::vector<std::string> errors = tile_check::CheckTile(tile);

        std::fprintf(out, "    {\"type\": %d, \"expansion\": \"%s\", \"count\": %d, ", bp.canonical_type,
                     EXPANSION_NAMES[bp.expansion], bp.count);
        std::fprintf(out, "\"edges\": [\"%s\", \"%s\", \"%s\", \"%s\"], ", tile_check::EdgeName(tile.edge[0]),
                     tile_check::EdgeName(tile.edge[1]), tile_check::EdgeName(tile.edge[2]),
                     tile_check::EdgeName(tile.edge[3]));
        std::fprintf(out, "\"links\": ");
        WriteIntArray(out, links, 4);
        std::fprintf(out, ", \"fields\": ");
        WriteIntArray(out, fields, HALF_EDGE_COUNT);
        std::fprintf(out, ", \"field_count\": %d, \"inner_field\": %d, \"field_city_sides\": ", tile.field_count,
                     tile.innerField());
        WriteIntArray(out, city_sides, MAX_TILE_FIELDS);
        std::fprintf(out, ", \"marks\": ");
        WriteIntArray(out, marks, 4);
        std::fprintf(out, ", \"tile_marks\": %d, \"errors\": %d}%s\n", tile.tile_marks, static_cast<int>(errors.size()),
                     row + 1 < CANONICAL_TILE_TYPE_COUNT ? "," : "");
    }
    std::fprintf(out, "  ]\n}\n");
    std::fclose(out);
}

} // namespace

int main(int argc, char **argv) {
    const char *path = argc > 1 ? argv[1] : "tile_table.json";

    std::printf("牌表: %d 種牌, 共 %d 張\n", CANONICAL_TILE_TYPE_COUNT, PHYSICAL_TILE_COUNT);
    for (int expansion = 0; expansion < EXPANSION_COUNT; ++expansion) {
        int types = 0;
        for (const TileBlueprint &bp : all_tiles) types += bp.expansion == expansion ? 1 : 0;
        const int tiles = tileCountIn(expansionBit(static_cast<Expansion>(expansion)));
        std::printf("  %-17s %3d 種  %3d / %3d 張%s\n", EXPANSION_NAMES[expansion], types, tiles,
                    OFFICIAL_TILE_COUNTS[expansion],
                    tiles == 0 ? "  (還沒填)" : tiles == OFFICIAL_TILE_COUNTS[expansion] ? "" : "  <-- 張數不對");
    }

    const std::vector<std::string> errors = tile_check::CheckTileTable();
    std::printf("\n錯誤: %d\n", static_cast<int>(errors.size()));
    for (const std::string &error : errors) std::printf("  %s\n", error.c_str());

    const std::vector<std::string> incomplete = tile_check::IncompleteExpansions();
    std::printf("\n未完成的擴充: %d\n", static_cast<int>(incomplete.size()));
    for (const std::string &problem : incomplete) std::printf("  %s\n", problem.c_str());

    const auto conflicts = tile_check::TileLookConflicts();
    std::printf("\n觀測分不出農田的牌對: %d\n", static_cast<int>(conflicts.size()));
    for (const auto &conflict : conflicts) {
        std::printf("  type %d 與 type %d (其中一種要在 carcassonne_test.cc 的 kHiddenFieldTypes 裡)\n",
                    conflict.first, conflict.second);
    }

    WriteJson(path);
    std::printf("\n寫出 %s\n", path);
    return errors.empty() ? 0 : 1;
}
