import Foundation

/// Which app build this is: the git commit scripts/build.sh stamped into Resources/build.json. Recordings save it,
/// so a frozen test can name the exact code that made every call.
struct BuildInfo: Codable, Sendable {
    var commit: String
    var dirty: Bool  // built with uncommitted changes
    var built: String  // ISO 8601, UTC

    static let current: BuildInfo? = Bundle.main.url(forResource: "build", withExtension: "json")
        .flatMap { try? Data(contentsOf: $0) }
        .flatMap { try? JSONDecoder().decode(BuildInfo.self, from: $0) }

    /// "abc1234", "abc1234+changes", or "unknown".
    var short: String {
        commit == "unknown" ? commit : String(commit.prefix(7)) + (dirty ? "+changes" : "")
    }
}
