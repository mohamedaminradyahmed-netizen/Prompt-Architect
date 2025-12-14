
import { ObjectStore } from '../storage/objectStore';

describe('ObjectStore', () => {
  it('should instantiate correctly', () => {
    const store = new ObjectStore('my-bucket');
    expect(store).toBeDefined();
  });
});
