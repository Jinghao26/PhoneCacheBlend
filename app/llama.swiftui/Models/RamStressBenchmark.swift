import Darwin
import Foundation
import os

/// Passage length profile for RAM stress ramp.
enum RamStressScale: String, CaseIterable, Identifiable {
  case simple = "Simple_test (~16–112 tok)"
  case wikiScale = "WikiMQA (~560 tok)"

  var id: String { rawValue }
}

/// Generates unique passage bodies for cumulative RAM preload.
enum RamStressPassages {
  /// Representative WikiMQA passage (~2.4k chars; ~560 tokens on Qwen).
  static let wikiMQARepresentativeBody = """
    march 1666, she married francis rakoczi, with whom she had three children : gyorgy, born in 1667, who died in infancy ; julianna, born in 1672 ; and ferenc ( commonly known as francis rakoczi ii ), born in 1676. on june 8, 1676, not long after francis ii's birth, the elder francis died. the widowed ilona requested guardianship of her children and was granted it, against the advice of emperor leopold i's advisers and against francis i's will. in this way she also retained control over the vast rakoczi estates, which included among them the castles of regec, sarospatak, makovica, and munkacs. in 1682 she married imre thokoly and became an active partner in her second husband's kuruc uprising against the habsburgs. defense of munkacs ( palanok ) castle after their defeat at the 1683 battle of vienna, both the ottoman forces and thokoly's allied kuruc fighters had no choice but to retreat, and thokoly quickly lost one rakoczi castle after another. at the end of 1685, the imperial army surrounded the last remaining stronghold, munkacs castle in today's ukraine. ilona zrinyi alone defended the castle for three years ( 1685 – 1688 ) against the forces of general antonio caraffa. internment, exile and death after the recapture of buda, the situation became untenable, and on 17 january 1688, ilona had no choice but to surrender the castle, with the understanding that the defenders would receive amnesty from the emperor, and that the rakoczi estates would remain in her children's name. under this agreement, she and her children traveled immediately to vienna, where in violation of the pact the children were taken from her. ilona lived until 1691 in the convent of the ursulines, where her daughter julianna was also raised. her son francis was immediately taken to the jesuit school in neuhaus. at the time, her husband, thokoly, was still fighting with his kuruc rebels against the habsburg army in upper hungary. when habsburg general heisler was captured by thokoly, a prisoner exchange was arranged, and ilona joined her husband in transylvania. in 1699, however, after the treaty of karlowitz was signed, both spouses, having found themselves on the losing side, had to go into exile in the ottoman empire. the countess lived in galata, district of constantinople, and later in izmit, where she died on 18 february 1703.
    """

  static func passageText(index: Int, scale: RamStressScale) -> String {
    let base: String
    switch scale {
    case .simple:
      let passages = QualityPassage.allCases
      base = passages[index % passages.count].text
      if index < passages.count {
        return base
      }
    case .wikiScale:
      base = wikiMQARepresentativeBody
    }
    return base + "\n\n[ram-stress slot \(index + 1)]"
  }

  static func passages(count: Int, scale: RamStressScale) -> [String] {
    guard count > 0 else { return [] }
    return (0..<count).map { passageText(index: $0, scale: scale) }
  }
}

struct RamKvBlobStats {
  let chunkEntries: Int
  let chunkBytes: Int
  let chunkTokens: Int
  let labelEntries: Int
  let labelBytes: Int
  let labelTokens: Int
  let diskEntries: Int
  let residentBytes: UInt64?
  let availableBytes: UInt64?

  var totalRamBytes: Int { chunkBytes + labelBytes }

  var avgChunkBytes: Int {
    chunkEntries > 0 ? chunkBytes / chunkEntries : 0
  }
}

enum RamStressFormat {
  static func bytes(_ value: Int) -> String {
    bytes(UInt64(value))
  }

  static func bytes(_ value: UInt64) -> String {
    let d = Double(value)
    if d >= 1_073_741_824 {
      return String(format: "%.2f GB", d / 1_073_741_824)
    }
    if d >= 1_048_576 {
      return String(format: "%.1f MB", d / 1_048_576)
    }
    if d >= 1024 {
      return String(format: "%.1f KB", d / 1024)
    }
    return "\(value) B"
  }
}

enum ProcessMemory {
  static func residentBytes() -> UInt64? {
    var info = mach_task_basic_info()
    var count = mach_msg_type_number_t(
      MemoryLayout<mach_task_basic_info>.size / MemoryLayout<natural_t>.size
    )
    let result = withUnsafeMutablePointer(to: &info) {
      $0.withMemoryRebound(to: integer_t.self, capacity: Int(count)) {
        task_info(mach_task_self_, task_flavor_t(MACH_TASK_BASIC_INFO), $0, &count)
      }
    }
    guard result == KERN_SUCCESS else { return nil }
    return UInt64(info.resident_size)
  }

  /// iOS jetsam headroom (nil on older SDK paths).
  static func availableBytes() -> UInt64? {
    if #available(iOS 13.0, *) {
      return UInt64(os_proc_available_memory())
    }
    return nil
  }
}
